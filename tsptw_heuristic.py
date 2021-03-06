# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import threading
from collections import OrderedDict
import copy
import time
import getopt, sys
import _thread
import random


# TODO avoid clone with 2 paths

# %%
NOT_FEASIBLE = -1
DEPOT = 0
N = None
graph = None
windows = None
opened = None
closed = None
departure = None
starts = None
path = None
visited = None
K = 0
P = 3
neighbors = {}
n_sucessors = {}
n_predecessors = {}


# %%
class Node:
    next = None
    prev = None
    value = None
    position = None
    def __init__(self, next, prev, value):
        self.prev, self.next, self.value = next, prev, value
    def __str__(self):
        return f"value: {self.value} prev: { '-' if self.prev == None else self.prev.value } next: { 'n' if self.next == None else self.next.value }"

class Path:
    path = None
    nodes = {}
    size = 0
    head = None
    tail = None
    not_visited = None
    starts = None
        
    def __init__(self):
        self.nodes = {  i: Node(None, None, i) for i in range(N) }

    def initialize_not_visited(self, nodes):
        self.not_visited = OrderedDict([(i,i) for i in nodes])

    def add_all(self, list):
        for node in list:
            self.add(node)

    def add(self, value, dry=False):
        node = self.nodes[value]
        
        if self.head == None:
            self.head, self.tail = node, node
        else:
            self.head.next = node
            node.prev = self.head
            self.head = node
        node.position = self.size
        self.size += 1
        if value != DEPOT and not dry:
            del self.not_visited[value]

    def get_head(self):
        node = self.tail
        while node != None and node.next != None:
            node = node.next
        return node

    def remove(self, value):
        if (value == DEPOT):
            raise NameError("cant remove the depot")
        if ( value in self.not_visited):
            raise NameError("trying to remove a node that does not exist in the path")
        node = self.nodes[value]
        prev = node.prev
        
        if prev == None:
            raise NameError("cant remove the depot")
        elif node.next == None:
            prev.next = None
            self.head = node.prev
        else:
            prev = node.prev
            next = node.next
            prev.next, next.prev = next, prev
        node.prev, node.next = None, None
        self.not_visited[node.value] = node.value
        self.size -= 1
        return prev.value

    def iter_path(self):
        cnode = self.nodes.get(0)
        cnode = cnode.next
        while cnode != None:
            yield cnode.value
            cnode = cnode.next

    def insert(self, after, value):
        if(not value in self.not_visited.keys() and self.not_visited[value] == None):
            raise NameError("trying to add node that is already visited ",value)
        node = self.nodes[value]
        after_node = self.nodes[after]
        if after_node.next == None:
            after_node.next = node
            node.prev = after_node
            self.head = node
        else:
            temp = after_node.next
            after_node.next, node.prev, node.next, temp.prev = node, after_node, temp, node
        self.size += 1
        del self.not_visited[value]


    def remove_first(self):
        # the node in the position 0 is the DEPOT
        if self.tail != None and self.tail.next != None:
            node_to_remove = self.tail.next
            self.remove(node_to_remove.value)
    

    def get_next(self, value):
        node = self.nodes[value]
        if node.next != None:
            return node.next
        else:
            raise NameError("No next node")

    def get(self, value):
        if not value in self.nodes.keys():
            raise NameError("invalid node id")
        return self.nodes[value]        


    def pop(self):
        temp = self.head
        self.head = self.head.prev
        self.head.next = None
        temp.prev = None
        self.size -= 1

    def clone(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            p = Path()
            cnode = self.tail
            while cnode != None: 
                p.add(cnode.value)
                cnode = cnode.next
            return p
    
    def cycle(self):
        visited = [False for i in self.nodes.keys()]
        node = self.nodes.get(0)
        while node.next != None:
            if visited[node.value]:
                raise NameError("there is a cycle")
            visited[node.value] = True
            node = node.next

    def perturbation(self, level):
        random.seed()
        solution = list(path.iter_path())
        for _ in range(level):
            u = random.randint(1, len(solution)-1)
            v = random.randint(1, len(solution)-1)

            if u != v : path.switch(solution[u],solution[v])
        

    def switch(self, u, v):
        self.nodes[u],self.nodes[v] = self.nodes[u], self.nodes[v]
        

    def is_valid(self, debug=False):
        self.cycle()
        prev = 0
        arrival_time = 0
        for v in self.iter_path():
            if graph[prev][v] == NOT_FEASIBLE:
                return False
            arrival_time =  max( arrival_time + graph[prev][v], windows[v][0])
            if arrival_time > windows[v][1]:
                return False
            if debug:
                print(f"prev: {prev} curr {v} arr: {arrival_time} w: {windows[v]} g: {graph[prev][v]}")
            prev = v

        if arrival_time > windows[0][1]:
            return False
        return True
    
    def markspan(self):
        prev = 0
        arrival_time = 0
        for v in self.iter_path():
            arrival_time =  max( arrival_time + graph[prev][v], windows[v][0])
        return arrival_time

    def build_infeasible_solution(self):
        for v in self.not_visited:
            self.add(v)

    def calculate_departure_time(self):
        node = self.tail
        start = 0
        self.starts = [ -1 for _ in range(N)]
        self.starts[0] = 0
        while node != None:
            succ = node.next
            if succ != None:
                arrival_time =  start + (graph[node.value][succ.value] if succ != None else 0)
                if arrival_time <= windows[succ.value][1]:
                    if arrival_time < windows[succ.value][0]:
                        start = windows[succ.value][0]
                    else:
                        start = arrival_time
                    
                    self.starts[succ.value] =  start
                else:
                    self.starts[node.value] = -1
            node = succ

# %% [markdown]
# Remove arcs that are not feasible, if Ai + Cij <= Bj then arc Vi, Vj is not feasible
# 
# https://pubsonline.informs.org/doi/10.1287/opre.46.3.330

# %%
def remove_unfeasible_arcs(n):
    for u in range(n):
        for v in range(u+1, n):
            if windows[u][0] + graph[u][v] > windows[v][1]:
                graph[u][v] = NOT_FEASIBLE
            if windows[v][0] + graph[v][u] > windows[u][1]:
                graph[v][u] = NOT_FEASIBLE


# %%
def time_windows_proximity(vi, vj):
    return min(windows[vj][1], windows[vi][1] + graph[vi][vj]) + max(windows[vj][0], windows[vi][0] + graph[vi][vj])


# %%
def pseudo_distance(vi, vj):
    ALPHA = 0.5
    return ALPHA*graph[vi][vj] + (1 - ALPHA)*time_windows_proximity(vi, vj)

# %%

def read_data(f):
    global N,graph,windows,departure,starts,path,opened,closed,visited, neighbors
    N = int(f.readline().strip())
    graph = [ [0 for _ in range(N)] for _ in range(N) ]
    windows = [ (0,0,0) for i in range(N)]
    departure = [-1 for _ in range(N)]
    starts = [-1 for _ in range(N)]
    path = Path()
    opened = [0 for _ in range(N)]
    closed = [0 for _ in range(N)]
    visited = [False for _ in range(N)]
    neighbors = {}

    for i in range(N):
        line = f.readline().strip().split(" ")
        for j in range(N):
            graph[i][j] = float(line[j])
    for i in range(N):
        line = f.readline().strip().split(" ")
        firstn = int(line[0])
        secondn = None
        for j in range(1,len(line)):
            if line[j] != '':
                secondn = int(line[j])
                break
        windows[i] = (firstn, secondn, i, secondn-firstn)
        opened[i] = firstn
        closed[i] = secondn

# %%
def is_insertion_feasible(vi, v):
    path.insert(vi,v)
    is_feasible = path.is_valid()
    path.remove(v)
    return is_feasible


# %%
def calculate_departure_time():
    global starts
    node = path.tail
    start = 0
    starts[0] = 0
    while node != None:
        succ = node.next
        if succ != None:
            arrival_time =  start + (graph[node.value][succ.value] if succ != None else 0)
            if arrival_time <= windows[succ.value][1]:
                if arrival_time < windows[succ.value][0]:
                    start = windows[succ.value][0]
                else:
                    start = arrival_time
                
                starts[succ.value] =  start
            else:
                starts[node.value] = -1
        node = succ

# %%
def sort_by_tw_ascending():
    global windows
    sorted_windows = sorted(windows[1:], key=lambda x: x[3])
    return sorted_windows

# %%
def construct_first_path():
    def arrives_in_time(arrival_time, w):
        return arrival_time <= w[1]

    sorted_windows = sort_by_tw_ascending()
    current_node_index = -1
    dep_time = graph[0][0]
    solution = [ 0 ]
    while current_node_index < len(sorted_windows) - 1:
        node = solution[-1]
        next_node_window = sorted_windows[current_node_index + 1]
        arrival_time = dep_time + graph[node][next_node_window[2]]
        can_arrive = arrives_in_time(arrival_time, next_node_window)
        if can_arrive:
            solution.append(next_node_window[2])
            visited[next_node_window[2]] = True # Mark as visited

            if arrival_time < next_node_window[0]:
                start = next_node_window[0]
            else:
                start = arrival_time
            dep_time = start + graph[node][next_node_window[2]]
        current_node_index += 1
    path.initialize_not_visited([ x[2] for x in sorted_windows ])
    path.add_all(solution)


# %%
def calculate_p_neighboors():
     for not_visited_node in path.not_visited.values():
        path_node = path.tail
        distances = []
        while path_node != None:
            # create a tuple of form (pseudo_distance, node_id) and add it to the distances list if it's a feasible arc
            d = time_windows_proximity(path_node.value, not_visited_node) if graph[path_node.value][not_visited_node]  != NOT_FEASIBLE else NOT_FEASIBLE
            if d != NOT_FEASIBLE:
                distances.append((d, path_node.value))
            path_node = path_node.next
        distances.sort()
        #print("node", not_visited_node, distances)
        neighbors[not_visited_node] = distances

def calculate_distances(distance_function, results):
    for i in range(N):
        path_node = path.tail
        distances = []
        while path_node != None:

            d = distance_function(path_node.value, i)
            if d != NOT_FEASIBLE:
                distances.append((d, path_node.value))
            path_node = path_node.next
        distances.sort()
        #print("node", not_visited_node, distances)
        results[i] = distances

def calculate_succesors_and_predecessors():
    calculate_distances(time_windows_proximity, n_sucessors)
    calculate_distances(pseudo_distance, n_predecessors)

# %%
def insert_remaining_nodes():
    calculate_p_neighboors()
    calculate_departure_time()
    not_visited_copy = path.not_visited.copy()
    for not_visited_node in not_visited_copy:
        #print("trying to insert " , not_visited_node)
        for (_, neighbor) in neighbors[not_visited_node][:]:
            if is_insertion_feasible(neighbor, not_visited_node):
                insert_node(neighbor, not_visited_node)
                #print("inserted", neighbor, not_visited_node)
                break
# %%
def insert_node(vi, v):
    path.insert(vi, v)
    calculate_p_neighboors()
    calculate_departure_time()

# %%

def type_one(not_visited_node, vin, vjn):
    global path
    inserted  = False
    i,j,k=0,0,0
    while not inserted and i < P:
        vi = vin[not_visited_node][i][1]
        while not inserted and j < P:
            vj = vjn[not_visited_node][i][1]
            while not inserted and k < P:
                vinext = path.get_next(vi).value
                vjnext = path.get_next(vj).value
                vk = n_sucessors[vinext][k][1]

               
                if vinext != vi and vinext != vj and vi != vj and vk != vi and vk != vj:
                    print(vi, vj, vinext, vjnext, vk)
                    cpath = path.clone(deep=True)
                    cpath.remove(vinext)
                    cpath.insert(vi, not_visited_node)
                    cpath.remove(vjnext)
                    cpath.insert(vj, vinext)
                    cpath.insert(vk, vjnext)

                    if cpath.is_valid():
                        print("it worked")
                        path = cpath
                        calculate_p_neighboors()
                        calculate_departure_time()
                        inserted = True
                    else:
                        print("not valid")
                k+=1
            j+=1
        i+=1
    return inserted

def try_to_insert_with_geni():
    global path    
    calculate_succesors_and_predecessors()
    calculate_departure_time()
    not_visited_copy = path.not_visited.copy()
    for not_visited_node in not_visited_copy:
        type_one(not_visited_node, n_sucessors, n_sucessors)

def unstring_and_string_3nodes():
    '''
    Tested removing 2,3 nodes from the actual solution to find a better solution
    similar to k-opt but removing nodes trying to find a better solution instead of reversing subpaths
    '''
    global path

    for i in range(1, N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                    if (i not in path.not_visited and 
                        j not in path.not_visited and 
                        k not in path.not_visited and
                        i != j and j != k and k != i):

                        cpath = path.clone(deep=True)
                        path.remove(i)
                        path.remove(j)
                        path.remove(k)

                        insert_remaining_nodes()

                        if path.size == N:
                            return
                        # found a better solution
                        if path.is_valid() and path.size > cpath.size:
                        
                            unstring_and_string_3nodes()
                            return
                        path = cpath #restore original path


# %%
def quit_function():
    # print to stderr, unbuffered in Python 2.
    print('Took too long trying to create a feasible solution')
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    _thread.interrupt_main()
            

def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


# %%
def post_optimization():
    for v in path.iter_path():
        if v not in path.not_visited and v != DEPOT:
            csize = path.size
            after_node = path.remove(v)
            try_to_insert_with_geni()
            if path.size < csize:
                path.insert(after_node, v)
            


# %%
def insertion_heuristic(f, debug=False):


    read_data(f)
    remove_unfeasible_arcs(N)
    construct_first_path()
    insert_remaining_nodes()
    try_to_insert_with_geni()
    unstring_and_string_3nodes()
    post_optimization()

    if debug:
        print(f"{ 'EUREKA' if path.size == N else 'INCOMPLETE:'+ str(N - path.size)} V:{path.is_valid()} M: {path.markspan()}")
    print(" ".join([str(x) for x in path.iter_path()]))


    return path, path.size == N

# %%


    
def multiple_set_execution():
    base = "./checker/SolomonPotvinBengio/"
    a = [(201,4),(202,4),(203,4),(204,3),(205,4),(206,4),(207,4),(208,3)]
    #a = [(201, 1)]
    solved, total = 0, 0
    avg_time = 0
    for configuration in a:

        for i in range(1,configuration[1]+1):
            filename = f"{base}rc_{configuration[0]}.{i}.txt"

            with open(filename) as f:
                print(filename)
                start = time.time()
                sol, s = insertion_heuristic(f, debug=True)
                solved += 1 if s else 0
                if not s:
                    s += 1 if vns() else 0
                end = time.time()
                print("total time", end - start)
                avg_time += end-start
                total+=1
            #with open("./checker/SolomonPotvinBengio/rc_201.1.txt") as f:
    print(f"solved {solved} problems of {total} in avg time: {avg_time/total}")




def vns(debug=False):
    global path
    level = 1
    ALPHA = 8
    BETA = 3
    T =  100
    t = 0
    #if len(path.not_visited) > 0:
    #    path.build_infeasible_solution()
    while not path.size == N and t < T:
        path.perturbation(random.randint(1, BETA))

        while not path.size == N and level < ALPHA:
            cpath = path.clone(deep=True)
            path.perturbation(level)
            insert_remaining_nodes()
            if path.size > cpath.size:
                level = 1
            else:
                path = cpath
                level += 1
        t += 1

    print(" ".join([str(x) for x in path.iter_path()]))
    return path.size == N



def main():

    argument_list = sys.argv[1:]
    
    # Options
    options = "t:f:o:-d"

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argument_list, options)
        filename,t = None, None
        debug = False
        # checking each argument
        for current_argument, current_value in arguments:
            if current_argument == "-t":
                t = int(current_value)
            if current_argument == "-f":
                filename = current_value
            if current_argument == "-d":
                debug = True
        if filename == None or t == None:
            raise ValueError("Please pass the filename and time as parameter of the script")

        with open(filename) as f:
            start = time.time()
            timer = threading.Timer(t, quit_function, args=[])
            timer.start()
            try:
                _, solved = insertion_heuristic(f, debug=debug)
                if not solved:
                    vns()
            finally:
                timer.cancel()
            end = time.time()
            if debug:
                print("total time: ", end - start)

    except getopt.error as err:
        print (str(err))


# %%

if __name__ == '__main__':
    #main()
    multiple_set_execution()


