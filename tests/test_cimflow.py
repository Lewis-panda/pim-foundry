"""Instruction/layout checks; native timing is opt-in and never arithmetic proof."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from pim_foundry.cimflow import compile_gemv, compile_matmul, execute, interpret, verify


class CimflowLoweringTests(unittest.TestCase):
    def test_signed_int8_and_int32_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"case"
            compile_gemv([-128,127,-1], [[-128,127],[127,-128],[-1,-128]],path,tile_k=8,tile_n=8)
            self.assertEqual(interpret(path)["output"], [[32514,-32384]])

    def test_one_hot_exposes_weight_transpose_and_padded_stride(self):
        rows, width = 17,19
        inputs = [[int(i==j) for j in range(rows)] for i in (0,8,15,16)]
        weights = [[(r*19+c)%251-125 for c in range(width)] for r in range(rows)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"case"
            manifest=compile_matmul(inputs,weights,path)
            self.assertEqual(interpret(path)["output"],[weights[i] for i in (0,8,15,16)])
            self.assertEqual(manifest["padded_shape"],{"k":32,"n":32})
            self.assertEqual(manifest["cim_mvm_count"],16)

    def test_signed_boundary_with_multiple_k_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"case"
            compile_gemv([-128]*128,[[-128,127]]*128,path)
            self.assertEqual(interpret(path)["output"],[[2097152,-2080768]])

    def test_seeded_ragged_shapes_all_tile_options(self):
        rng=random.Random(20261015)
        with tempfile.TemporaryDirectory() as tmp:
            for index,(m,k,n,tk,tn) in enumerate([(1,1,1,8,8),(2,15,17,16,8),(3,17,15,8,16),
                                               (4,33,35,32,32),(2,65,67,64,64),(16,7,9,16,16)]):
                a=[[rng.randrange(-128,128) for _ in range(k)] for _ in range(m)]
                b=[[rng.randrange(-128,128) for _ in range(n)] for _ in range(k)]
                expected=[[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
                path=Path(tmp)/str(index)
                compile_matmul(a,b,path,tile_k=tk,tile_n=tn)
                with self.subTest(shape=(m,k,n),tile=(tk,tn)):
                    self.assertEqual(interpret(path)["output"],expected)

    def test_zero_and_repeated_inputs_have_fresh_accumulators(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"case"
            compile_matmul([[0]*17,[1]*17,[1]*17],[[2]*17]*17,path)
            self.assertEqual(interpret(path)["output"],[[0]*17,[34]*17,[34]*17])

    def test_instruction_interpreter_does_not_read_expected_or_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"case"
            compile_gemv([2,3],[[4],[5]],path)
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["expected_output"]=[[99999]]
            (path/"manifest.json").write_text(json.dumps(manifest))
            (path/"workload.json").write_text("{}")
            self.assertEqual(interpret(path,verify_hashes=False)["output"],[[23]])

    def test_corrupted_weight_image_changes_instruction_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"case"
            manifest=compile_gemv([2],[[4]],path)
            instructions=json.loads((path/"instructions.json").read_text())["0"]
            copy_index=next(i for i,ins in enumerate(instructions) if ins["opcode"]==48)
            weight_base=instructions[copy_index-3]["imm"]
            data=bytearray((path/"memory.bin").read_bytes())
            data[weight_base-manifest["memory"]["staging_base"]]=5
            (path/"memory.bin").write_bytes(data)
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                verify(path)
            self.assertEqual(interpret(path,verify_hashes=False)["output"],[[10]])

    def test_invalid_shapes_types_precision_and_capacity_fail(self):
        cases=[([],[[1]]),([[1,2]],[[1]]),([[1],[1,2]],[[1]]),([[True]],[[1]]),
               ([[1.0]],[[1]]),([[128]],[[1]]),([[-129]],[[1]]),([[1]]*17,[[1]]),
               ([[1]*129],[[1]]*129),([[1]],[[1]*129])]
        with tempfile.TemporaryDirectory() as tmp:
            for index,(a,b) in enumerate(cases):
                with self.subTest(index=index),self.assertRaises(ValueError):
                    compile_matmul(a,b,Path(tmp)/str(index))
            for tile in (True,0,7,128,16.0):
                with self.subTest(tile=tile),self.assertRaises(ValueError):
                    compile_gemv([1],[[1]],Path(tmp)/"bad",tile_k=tile)

    def test_no_overwrite_and_no_unsafe_artifact_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"case"
            compile_gemv([1],[[1]],path)
            with self.assertRaises(FileExistsError):
                compile_gemv([1],[[1]],path)
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["files"]["../outside"]="bad"
            (path/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"coverage"):
                verify(path)

    def test_illegal_instruction_and_out_of_bounds_memory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"case"
            compile_gemv([1],[[1]],path)
            original=json.loads((path/"instructions.json").read_text())
            invalid=copy.deepcopy(original)
            invalid["0"].append({"opcode":999})
            (path/"instructions.json").write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,"opcode"):
                interpret(path,verify_hashes=False)
            invalid=copy.deepcopy(original)
            copy_index=next(i for i,ins in enumerate(invalid["0"]) if ins["opcode"]==48)
            invalid["0"][copy_index-3]["imm"]=10**9
            (path/"instructions.json").write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,"bounds"):
                interpret(path,verify_hashes=False)

    def test_evidence_separates_native_timing_and_functional_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=compile_gemv([1],[[1]],Path(tmp)/"case")
            self.assertFalse(manifest["native_numerical"])
            self.assertIn("host_to_staging",manifest["excluded_costs"])
            self.assertIn("not_paper_chip",manifest["profile_provenance"])


@unittest.skipUnless(os.environ.get("PIM_FOUNDRY_CIMFLOW_SIMULATOR"),"native simulator path not configured")
class CimflowNativeTests(unittest.TestCase):
    def test_ragged_native_execution_and_functional_lowering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"case"
            compile_matmul([[1]*17,[-1]*17],[[2]*19]*17,path)
            result=execute(path,Path(os.environ["PIM_FOUNDRY_CIMFLOW_SIMULATOR"]))
            self.assertEqual(result["output"],[[34]*19,[-34]*19])
            self.assertTrue(result["lowering_verified"])
            self.assertFalse(result["native_numerical"])
            self.assertGreater(result["native_timing_ns"],0)
            self.assertGreater(result["native_energy_pj"],0)
            with self.assertRaises(FileExistsError):
                execute(path,Path(os.environ["PIM_FOUNDRY_CIMFLOW_SIMULATOR"]))
