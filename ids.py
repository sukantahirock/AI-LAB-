from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)

def depth_limited_search(graph, src, target, depth, visited):
    visited.append(src)
    if src == target:
        return True, visited, 1 
    if depth == 0:
        return False, visited, 1 
    total_visited = 1 
    for neighbor in graph.graph[src]:
        if neighbor not in visited:  
            result, visited, neighbor_visited = depth_limited_search(graph, neighbor, target, depth - 1, visited)
            total_visited += neighbor_visited
            if result:
                return True, visited, total_visited
    return False, visited, total_visited

def iterative_deepening_search(graph, src, target, max_depth):
    total_visited = 0 
    for depth in range(max_depth + 1):
        result, visited, depth_visited = depth_limited_search(graph, src, target, depth, [])
        total_visited += depth_visited
        if result:
            return visited, total_visited
    return [], total_visited
def take_graph_input():
    vertices = int(input("Enter the number of vertices: "))
    g = Graph(vertices)
    while True:
        edge = input("Enter an edge or type 'ok' to finish: ")
        if edge.lower() == 'ok':
            break
        try:
            u, v = map(int, edge.split())
            g.addEdge(u, v)
        except ValueError:
            print("Invalid input. Please enter source and destination vertices separated by a space.")
    return g

g = take_graph_input()
"""
g = Graph(7)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 3)
g.addEdge(1, 4)
g.addEdge(2, 5)
g.addEdge(2, 6)
"""
src = 0
target = 6
max_depth = 3

path, total_visited = iterative_deepening_search(g, src, target, max_depth)

if path:
    print("Path found:", path)
    print("Total visited nodes:", total_visited)
else:
    print("Path not found.")
