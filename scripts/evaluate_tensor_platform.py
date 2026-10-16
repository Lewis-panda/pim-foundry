#!/usr/bin/env python3
"""Run/replay one actual n-D pipeline on four explicit heterogeneous platforms."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pim_foundry import application, unified, unified_examples, tensor_platform


def fixtures(directory):
    directory.mkdir(parents=True)
    graph = {"schema": "pim-foundry.tensor-graph.v2", "values": {}, "inputs": ["input"], "outputs": ["probabilities"],
             "initializers": {"w1": [[1,0,1,0],[0,1,0,1],[1,-1,0,1]], "w2": [[1,0],[0,1],[1,1],[-1,1]]}, "nodes": []}
    for name, shape, dtype in (("input", [2,2,3], "int8"), ("w1", [3,4], "int8"), ("hidden", [2,2,4], "int32"),
                               ("half", [2,2,4], "float16"), ("w2", [4,2], "float16"), ("projection", [2,2,2], "float16"),
                               ("logits", [2,2,2], "float32"), ("probabilities", [2,2,2], "float32")):
        graph["values"][name] = {"shape": shape, "dtype": dtype}
    for name, op, inputs, out, attrs in (("sram_linear", "matmul", ["input","w1"], "hidden", {}),
                                        ("precision", "cast", ["hidden"], "half", {"dtype":"float16"}),
                                        ("hbm_linear", "matmul", ["half","w2"], "projection", {}),
                                        ("logits_float", "cast", ["projection"], "logits", {"dtype":"float32"}),
                                        ("normalization", "softmax", ["logits"], "probabilities", {"axis":-1})):
        graph["nodes"].append({"name":name,"op":op,"inputs":inputs,"output":out,"attrs":attrs})
    inputs = np.arange(-6,6,dtype=np.int8).reshape(2,2,3)
    # Independent exact integer loops, explicit half storage, scalar exp oracle.
    reference = []
    for batch in inputs.tolist():
        rows=[]
        for row in batch:
            hidden=[sum(row[k]*graph["initializers"]["w1"][k][j] for k in range(3)) for j in range(4)]
            projection=[sum(hidden[k]*graph["initializers"]["w2"][k][j] for k in range(4)) for j in range(2)]
            exps=[math.exp(x-max(projection)) for x in projection]
            rows.append([x/sum(exps) for x in exps])
        reference.append(rows)
    unified.save(directory/"graph.json",graph)
    np.savez(directory/"inputs.npz",input=inputs)
    np.savez(directory/"reference.npz",probabilities=np.asarray(reference,dtype=np.float32))
    app={"schema":application.APP_SCHEMA,"id":"batched-mixed-cim-probabilities","adapter":"tensor-graph/v2",
         "assets":{"graph":application.asset(directory/"graph.json","model","pim-foundry.tensor-graph.v2"),
                   "inputs":application.asset(directory/"inputs.npz","inputs","npz"),
                   "reference":application.asset(directory/"reference.npz","quality","npz")},
         "parameters":{},"scenario":{"kind":"tensor_requests","arrivals_ns":[0,0,50],"max_inflight":2,"initial_state":"warm"},
         "quality":{"kind":"numerical_equivalence","reference":"independent_integer_loops_and_scalar_exp/v1","atol":1e-7,"rtol":1e-6}}
    unified.save(directory/"application.json",app)
    return graph, app


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    graph,app=fixtures(root/"fixtures");cases=[]
    for name,kinds in (("sram-hbm-gpu",("sram","hbm","gpu")),("hbm-gpu",("hbm","gpu")),("hbm-npu",("hbm","npu")),("cpu",("cpu",))):
        platform=tensor_platform.default_platform() if kinds == ("cpu",) else unified_examples.platform(kinds,root=args.project_root)
        software=next(k for k in kinds if k not in ("sram","hbm"))
        placement={n["name"]:software for n in graph["nodes"]}
        if "sram" in kinds:placement["sram_linear"]="sram"
        if "hbm" in kinds:placement["hbm_linear"]="hbm"
        experiment={"schema":application.EXPERIMENT_SCHEMA,"platform":platform,"mapping":{"strategy":"explicit","placement":placement},"variant":{}}
        unified.save(root/(name+".json"),experiment)
        result=application.run(root/"fixtures/application.json",root/(name+".json"),root/name)
        verified=application.verify(root/name)
        replay=application.replay(root/name,root/(name+"-replay"))
        cases.append({"case":name,"status":result["status"],"verified":verified["passed"],"replay":replay["passed"],"metrics":result["metrics"]})
    bad=np.full((2,2,2),0.5,dtype=np.float32);np.savez(root/"fixtures/wrong.npz",probabilities=bad)
    app["assets"]["reference"]=application.asset(root/"fixtures/wrong.npz","quality","npz")
    unified.save(root/"fixtures/wrong-application.json",app)
    negative=application.run(root/"fixtures/wrong-application.json",root/"cpu.json",root/"wrong-reference")
    summary={"schema":"pim-foundry.tensor-platform-acceptance.v1","cases":cases,
             "wrong_reference_rejected":negative["status"]=="quality_failed",
             "physical_hardware_performance_claim":False,
             "passed":all(c["status"]=="completed" and c["verified"] and c["replay"] for c in cases) and negative["status"]=="quality_failed"}
    unified.save(root/"summary.json",summary);print(json.dumps(summary,indent=2));return 0 if summary["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
