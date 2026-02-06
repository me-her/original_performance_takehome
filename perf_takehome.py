"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict, deque
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.current_bundle = defaultdict(list)
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def flush_bundle(self):
        if self.current_bundle:
            self.instrs.append(dict(self.current_bundle))
            self.current_bundle = defaultdict(list)

    def add(self, engine, slot):
        if len(self.current_bundle[engine]) >= SLOT_LIMITS[engine]:
            self.flush_bundle()
        self.current_bundle[engine].append(slot)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # --- Metadata & Constants ---
        metadata_addrs = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        meta = {name: self.alloc_scratch(name) for name in metadata_addrs}
        curr_ptr = self.alloc_scratch("init_ptr")
        for i, name in enumerate(metadata_addrs):
            self.add("load", ("const", curr_ptr, i))
            self.flush_bundle()
            self.add("load", ("load", meta[name], curr_ptr))
        
        NB = 16 if batch_size >= 128 else 4
        
        stride = VLEN * NB
        stride_const = self.scratch_const(stride)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        zero_const = self.scratch_const(0)
        
        # Broadcasts
        one_vec = self.alloc_scratch("one_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        self.add("valu", ("vbroadcast", one_vec, one_const))
        self.add("valu", ("vbroadcast", two_vec, two_const))
        self.add("valu", ("vbroadcast", zero_vec, zero_const))
        self.add("valu", ("vbroadcast", n_nodes_vec, meta["n_nodes"]))
        
        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1, f"h_{hi}_v1")
            c3 = self.scratch_const(val3, f"h_{hi}_v3")
            mc_val = None
            if hi in [0, 2, 4]:
                mc_val = (1 << val3) + 1
            mc = self.scratch_const(mc_val, f"h_{hi}_mc") if mc_val else None

            self.flush_bundle()
            bc1 = self.alloc_scratch(f"h_{hi}_c1", VLEN)
            bc3 = self.alloc_scratch(f"h_{hi}_c3", VLEN)
            bmc = self.alloc_scratch(f"h_{hi}_mc_vec", VLEN) if mc_val else None

            self.add("valu", ("vbroadcast", bc1, c1))
            self.add("valu", ("vbroadcast", bc3, c3))
            if bmc:
                self.add("valu", ("vbroadcast", bmc, mc))
            hash_const_vecs.append((bc1, bc3, bmc))
        
        # Scratch
        idx = [self.alloc_scratch(f"idx{b}", VLEN) for b in range(NB)]
        val = [self.alloc_scratch(f"val{b}", VLEN) for b in range(NB)]
        node_A = [self.alloc_scratch(f"nodeA{b}", VLEN) for b in range(NB)]
        node_B = [self.alloc_scratch(f"nodeB{b}", VLEN) for b in range(NB)]
        
        tmp1 = [self.alloc_scratch(f"tmp1_{b}", VLEN) for b in range(NB)]
        tmp2 = [self.alloc_scratch(f"tmp2_{b}", VLEN) for b in range(NB)]
        addr = [self.alloc_scratch(f"addr{b}", VLEN) for b in range(NB)]
        
        ptr_idx = [self.alloc_scratch(f"ptr_i{b}") for b in range(NB)]
        ptr_val = [self.alloc_scratch(f"ptr_v{b}") for b in range(NB)]
        
        self.add("flow", ("pause",))
        self.flush_bundle()

        block_offsets = [self.scratch_const(b * VLEN) for b in range(NB)]
        self.flush_bundle()

        num_iters = (batch_size + stride - 1) // stride
        
        # Init Pointers
        for b in range(NB):
            self.add("alu", ("+", ptr_idx[b], meta["inp_indices_p"], block_offsets[b]))
            self.add("alu", ("+", ptr_val[b], meta["inp_values_p"], block_offsets[b]))
        self.flush_bundle()

        # Main Loop
        delay_per_block = 0
        
        # --- Stream Logic ---
        class StreamBuilder:
            def __init__(self):
                self.bundles = []
                self.current = defaultdict(list)
            
            def add(self, engine, slot):
                if len(self.current[engine]) >= SLOT_LIMITS[engine]:
                    self.flush()
                self.current[engine].append(slot)
            
            def flush(self):
                if self.current:
                    self.bundles.append(dict(self.current))
                    self.current = defaultdict(list)
            
            def finish(self):
                self.flush()
                return deque(self.bundles)

        def make_stream(b, delay=0):
            sb = StreamBuilder()
            
            # Delay start to stagger blocks and maximize overlap
            for _ in range(delay):
                sb.add("alu", ("+", ptr_idx[b], ptr_idx[b], zero_const))
                sb.flush()
            
            # 1. LOAD PHASE
            sb.add("load", ("vload", idx[b], ptr_idx[b]))
            sb.add("load", ("vload", val[b], ptr_val[b]))
            sb.flush()
            
            # Initial Gather (L0) - Standard
            for vi in range(VLEN):
                sb.add("alu", ("+", addr[b] + vi, meta["forest_values_p"], idx[b] + vi))
            sb.flush()
            for vi in range(0, VLEN, 2):
                sb.add("load", ("load", node_A[b] + vi, addr[b] + vi))
                sb.add("load", ("load", node_A[b] + vi + 1, addr[b] + vi + 1))
            sb.flush()
            
            # 2. ROUNDS
            for round_idx in range(rounds):
                node_curr = node_A[b] if round_idx % 2 == 0 else node_B[b]
                node_next = node_B[b] if round_idx % 2 == 0 else node_A[b]
                
                # XOR
                sb.add("valu", ("^", val[b], val[b], node_curr))
                sb.flush()
                
                # Hash Stages
                # Hash Stages
                for hi in range(6):
                    op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                    bc1, bc3, bmc = hash_const_vecs[hi]
                    
                    if bmc:
                        sb.add("valu", ("multiply_add", val[b], val[b], bmc, bc1))
                        sb.flush()
                    else:
                        sb.add("valu", (op1, tmp1[b], val[b], bc1))
                        sb.add("valu", (op3, tmp2[b], val[b], bc3))
                        sb.flush()
                        sb.add("valu", (op2, val[b], tmp1[b], tmp2[b]))
                        sb.flush()
                
                # Index Update
                # Optimize dependency: 
                # tmp1 = val & 1
                # idx = idx * 2 + 1 (parallel with tmp1)
                # idx = idx + tmp1
                
                sb.add("valu", ("&", tmp1[b], val[b], one_vec))
                sb.add("valu", ("multiply_add", idx[b], idx[b], two_vec, one_vec))
                sb.flush()
                
                sb.add("valu", ("+", idx[b], idx[b], tmp1[b]))
                sb.flush()
                
                # Wrap Check
                sb.add("valu", ("<", tmp1[b], idx[b], n_nodes_vec))
                sb.flush()
                # Use flow vselect to offload VALU unit
                sb.add("flow", ("vselect", idx[b], tmp1[b], idx[b], zero_vec))
                sb.flush()
                
                # Gather for Next Round (Standard)
                if round_idx < rounds - 1:
                    for vi in range(VLEN):
                        sb.add("alu", ("+", addr[b] + vi, meta["forest_values_p"], idx[b] + vi))
                    sb.flush()
                    for vi in range(0, VLEN, 2):
                        sb.add("load", ("load", node_next + vi, addr[b] + vi))
                        sb.add("load", ("load", node_next + vi + 1, addr[b] + vi + 1))
                    sb.flush()
            
            # 3. STORE PHASE
            sb.add("store", ("vstore", ptr_idx[b], idx[b]))
            sb.add("store", ("vstore", ptr_val[b], val[b]))
            sb.flush()
            
            # 4. UPDATE POINTERS
            sb.add("alu", ("+", ptr_idx[b], ptr_idx[b], stride_const))
            sb.add("alu", ("+", ptr_val[b], ptr_val[b], stride_const))
            sb.flush()
            
            return sb.finish()

        def merge_streams(streams):
            while any(streams):
                master = defaultdict(list)
                for stream in streams:
                    if not stream: continue
                    bundle = stream[0]
                    
                    take_moves = [] 
                    fully = True
                    for engine, ops in bundle.items():
                        limit = SLOT_LIMITS[engine]
                        curr = len(master[engine])
                        can = max(0, limit - curr)
                        count = len(ops)
                        if count > can:
                            take_moves.append((engine, can))
                            fully = False
                        else:
                            take_moves.append((engine, count))
                    
                    for engine, count in take_moves:
                        if count > 0:
                            master[engine].extend(bundle[engine][:count])
                            if count == len(bundle[engine]):
                                del bundle[engine]
                            else:
                                bundle[engine] = bundle[engine][count:]
                    
                    if fully and not bundle:
                        stream.popleft()
                    elif not bundle:
                         stream.popleft()
                
                if master:
                    self.instrs.append(dict(master))
                else: 
                     break 

        # Main Loop
        for it in range(num_iters):
            streams = []
            for b in range(NB):
                streams.append(make_stream(b, b * delay_per_block))
            
            merge_streams(streams)

        self.add("flow", ("pause",))
        self.flush_bundle()


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    
    # Run fully
    machine.run()
    
    # Check final
    final_ref = None
    for m in reference_kernel2(mem, value_trace):
        final_ref = m
        
    inp_values_p = final_ref[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == final_ref[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect final result"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        # Rounds=1 for debug, Batch=32, Prints=True
        do_kernel_test(10, 1, 32, prints=True)


if __name__ == "__main__":
    unittest.main()
