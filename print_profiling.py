import pstats
p = pstats.Stats("profiling")
p.sort_stats('tottime').print_stats()