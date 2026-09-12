from syn.synth import audit


def test_reject_readonly_memory_and_runtime_weight_multiplier():
    net = {
        "modules": {
            "linear": {
                "cells": {
                    "weights": {
                        "type": "$mem_v2",
                        "parameters": {"WR_PORTS": "0", "WIDTH": "1000", "SIZE": "100"},
                    },
                    "mul": {"type": "$mul", "parameters": {}},
                }
            }
        }
    }
    result = audit(net, ["linear"])
    assert not result["passed"] and len(result["violations"]) == 2


def test_allow_mutable_attention_storage_and_multiplication():
    net = {
        "modules": {
            "linear": {"cells": {}},
            "attention": {
                "cells": {
                    "kv": {
                        "type": "$mem_v2",
                        "parameters": {"WR_PORTS": "1", "WIDTH": "1000", "SIZE": "100"},
                    },
                    "mul": {"type": "$mul", "parameters": {}},
                }
            },
        }
    }
    assert audit(net, ["linear"])["passed"]
