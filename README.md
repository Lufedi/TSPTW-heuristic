### Description

This is an effort to implement an heuristic constructive algorithm to solve the travel salesman problem with time windows restrictions.

### How to run it


```
python tsptw_heuristic.py -f "./checker/SolomonPotvinBengio/rc_202.4.txt" -t 80   
```

**Options**

*Mandatory*
`-f` location of the input file i.e. `./checker/SolomonPotvinBengio/rc_201.1.txt`
`-t` maximum time in seconds allowed to run the script

*Optional*

`-d` debug, helps to log more information in the console about the status of the script



### Paper (under development)

[Link](https://github.com/Lufedi/TSPTW-heuristic/blob/main/Implementaci_n_de_TSPTW.pdf)


## Current results

| Case    | Solution | Time | Missing nodes | Makespan |
|----------|----------|--------|-----------|----------|
| rc_201.1 | Y       | 0.02   | 0         | 793.2431 |
| rc_201.2 | Y       | 0.18   | 0         | 843.095  |
| rc_201.3 | Y       | 0.14   | 0         | 962.070  |
| rc_201.4 | Y       | 0.17   | 0         | 1033.0   |
| rc_202.1 | Y       | 0.27   | 0         | 1217.4   |
| rc_202.2 | Y       | 0.007  | 0         | 427.182  |
| rc_202.3 | Y       | 0.33   | 0         | 921.53   |
| rc_202.4 | N       | 3.3    | 1         | 829.2    |
| rc_203.1 | Y       | 0.17   | 0         | 597.45   |
| rc_203.2 | Y       | 0.11   | 0         | 1021.44  |
| rc_203.3 | Y       | 1.53   | 0         | 1383.2   |
| rc_203.4 | Y       | 0.008  | 0         | 470.5    |
| rc_204.1 | N       | 72     | 1         | 1434.1   |
| rc_204.2 | Y       | 0.11   | 0         | 957.5    |
| rc_204.3 | Y       | 0.35   | 0         | 1028.5   |
| rc_205.1 | Y       | 0.008  | 0         | 449.1    |
| rc_205.2 | Y       | 0.24   | 0         | 951.49   |
| rc_205.3 | Y       | 0.33   | 0         | 112.6    |
| rc_205.4 | N       | 3,43   | 0         | 1152.8   |
| rc_206.1 | Y       | 0.000  | 0         | 851.782  |
| rc_206.2 | N       | 18     | 1         | 1205.62  |
| rc_206.3 | Y       | 0.32   | 0         | 877.2    |
| rc_206.4 | N       | 19     | 1         | 1020.3   |
| rc_207.1 | Y       | 0.22   | 0         | 1364.23  |
| rc_207.2 | Y       | 2.96   | 0         | 970.64   |
| rc_207.3 | Y       | 0.13   | 0         | 893.94   |
| rc_207.4 | Y       | 0.001  | 0         | 160.03   |
| rc_208.1 | Y       | 7.62   | 0         | 1347.76  |
| rc_208.2 | Y       | 0.07   | 0         | 834.7    |
| rc_207.3 | Y       | 0.20   | 0         | 1127.449 |
|          |          |        |           |          |
