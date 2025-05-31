import heapq
import matplotlib.pyplot as plt

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path

def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def distance(current, neighbor):
    return 1

def astar(start, goal, graph):
    open_set = PriorityQueue()
    open_set.put(start, 0)

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                if neighbor not in [i[1] for i in open_set.elements]:
                    open_set.put(neighbor, f_score[neighbor])

    return "No path found"

def visualize_path(graph, path, start, goal):
    grid_size = (max(node[0] for node in graph) + 1, max(node[1] for node in graph) + 1)
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, grid_size[0] - 0.5)
    ax.set_ylim(-0.5, grid_size[1] - 0.5)
    ax.set_xticks(range(grid_size[0]))
    ax.set_yticks(range(grid_size[1]))
    ax.grid(True)

    for node in graph:
        for neighbor in graph[node]:
            plt.plot([node[0], neighbor[0]], [node[1], neighbor[1]], 'gray')

    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'ro-')

    plt.plot(start[0], start[1], 'go')
    plt.plot(goal[0], goal[1], 'bo')

    plt.show()

# Example usage:
graph = {
    (0, 0): {(0, 1): 1, (1, 0): 1},
    (0, 1): {(0, 0): 1, (1, 1): 1},
    (1, 0): {(0, 0): 1, (1, 1): 1},
    (1, 1): {(0, 1): 1, (1, 0): 1}
}

start = (0, 0)
goal = (1, 1)

path = astar(start, goal, graph)
print("Path found:", path)

visualize_path(graph, path, start, goal)
