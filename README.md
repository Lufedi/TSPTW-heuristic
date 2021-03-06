#Description

This is an effort to implement an heuristic constructive algorithm to solve the travel salesman problem with time windows restrictions.

### How to run it


```
python tsptw_2.py -f "./checker/SolomonPotvinBengio/rc_202.4.txt" -t 80   
```

**Options**

*Mandatory*
`-f` location of the input file i.e. `./checker/SolomonPotvinBengio/rc_201.1.txt`
`-t` maximum time in seconds allowed to run the script

*Optional*

`-d` debug, helps to log more information in the console about the status of the script



### Paper (under development)
Plesae check the repo files, there you can find the PDF file with more details of the investigation process and results.


## Current results

| Case     | Solved   | Time   | Missing Nodes |
|----------|----------|--------|---------------|
| rc_201.1 | Yes       | 0.02   | 0             |
| rc_201.2 | Yes       | 0.18   | 0             |
| rc_201.3 | Yes       | 0.14   | 0             |
| rc_201.4 | Yes       | 0.17   | 0             |
| rc_202.1 | Yes       | 0.27   | 0             |
| rc_202.2 | Yes       | 0.007  | 0             |
| rc_202.3 | Yes       | 0.33   | 0         |
| rc_202.4 | No        | 3.3    | 1         |
| rc_203.1 | Yes       | 0.17   | 0         |
| rc_203.2 | Yes       | 0.11   | 0         |
| rc_203.3 | Yes       | 1.53   | 0         |
| rc_203.4 | Yes       | 0.008  | 0         |
| rc_204.1 | No        | 72     | 0         |
| rc_204.2 | Yes       | 0.11   | 0         |
| rc_204.3 | Yes       | 0.35   | 0         |
| rc_205.1 | Yes       | 0.008  | 0         |
| rc_205.2 | Yes       | 0.24   | 0         |
| rc_205.3 | Yes       | 0.33   | 0         |
| rc_205.4 | No        | 3,43   | 0         |
| rc_206.1 | Yes       | 0.000  | 0         |
| rc_206.2 | No        | 18     | 1         |
| rc_206.3 | Yes       | 0.32   | 0         |
| rc_206.4 | No        | 19     | 1         |
| rc_207.1 | Yes       | 0.22   | 0         |
| rc_207.2 | Yes       | 2.96   | 0         |
| rc_207.3 | Yes       | 0.13   | 0         |
| rc_207.4 | Yes       | 0.001  | 0         |
| rc_208.1 | Yes       | 7.62   | 0         |
| rc_208.2 | Yes       | 0.07   | 0         |
| rc_207.3 | Yes       | 0.20   | 0         |
