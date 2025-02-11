### Day12 diving deeper into profiler results

If you remember from day 11, we got our conflicts management kernel not performing as well as the base optim model. 

To further investigate, we can see ask ncu to give extra metrics such as gpu_dram_output and l1tex (l1 cache) throughput. 

We can see that our matmul conflict kernel  shows decreased dram throughput and increased l1 cache throughput.

```
  matmul_coal_optim
    ------------------------------------------------------ ----------- ------------
    Metric Name                                            Metric Unit Metric Value
    ------------------------------------------------------ ----------- ------------
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed           %        11.31
    l1tex__throughput.avg.pct_of_peak_sustained_active               %        89.43
    ------------------------------------------------------ ----------- ------------

  matmul_conflicts
    Section: Command line profiler metrics
    ------------------------------------------------------ ----------- ------------
    Metric Name                                            Metric Unit Metric Value
    ------------------------------------------------------ ----------- ------------
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed           %         8.72
    l1tex__throughput.avg.pct_of_peak_sustained_active               %        90.03
    ------------------------------------------------------ ----------- ------------

```

This indicates that more data is being cached, and global memory efficiency is reduced, however when running occupancy limits and registers per thread :

```
  matmul_coal_optim
    Section: Command line profiler metrics
    --------------------------------- --------------- ------------
    Metric Name                           Metric Unit Metric Value
    --------------------------------- --------------- ------------
    launch__occupancy_limit_registers           block            1
    launch__registers_per_thread      register/thread           41
    --------------------------------- --------------- ------------

  matmul_conflicts
    Section: Command line profiler metrics
    --------------------------------- --------------- ------------
    Metric Name                           Metric Unit Metric Value
    --------------------------------- --------------- ------------
    launch__occupancy_limit_registers           block            1
    launch__registers_per_thread      register/thread           36
    --------------------------------- --------------- ------------
```

We notice that matmul_conflicts uses fewer registers per thread (36), this means that the kernel is not suffering from high registery pressure. If it would, much higher number of registers per threads would be requested which would lead to spilling, so it is not the culprit here. 

