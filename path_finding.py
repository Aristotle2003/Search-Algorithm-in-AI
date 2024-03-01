import numpy as numpy
from queue import PriorityQueue
from utils.utils import PathPlanMode, Heuristic, cost, expand, visualize_expanded, visualize_path
import numpy as np

def uninformed_search(grid, start, goal, mode: PathPlanMode):
    frontier = [start]
    frontier_sizes = [len(frontier)]
    expanded = []
    reached = {start: None}

    while frontier:
        # Choose the current node: stack for DFS, queue for BFS
        if mode == PathPlanMode.DFS:
            current = frontier.pop()  # Removes and returns the last item
        else:  # BFS
            current = frontier.pop(0)  # Removes and returns the first item

        # Goal check
        if current == goal:
            # Build and return the path from start to goal
            path = []
            while current:
                path.append(current)
                current = reached[current]
            path.reverse()  # The path is constructed backwards from goal to start
            return path, expanded, frontier_sizes

        expanded.append(current)

        for next_cell in expand(grid, current):
            if next_cell not in reached:
                frontier.append(next_cell)
                reached[next_cell] = current

        frontier_sizes.append(len(frontier))
    # If the goal is not reached
    return [], expanded, frontier_sizes

def a_star(grid, start, goal, mode, heuristic, width):
    def heuristic_cost(heuristic, current, goal):
        if heuristic == Heuristic.MANHATTAN:
            return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
        elif heuristic == Heuristic.EUCLIDEAN:
            return numpy.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)

    frontier = PriorityQueue()
    frontier.put((0 + heuristic_cost(heuristic, start, goal), start))
    frontier_sizes = []
    expanded = []
    reached = {start: {"cost": 0, "parent": None}}  # Start cost is 0

    while not frontier.empty():
        current_priority, current = frontier.get()
        current_cost = reached[current]["cost"]
        frontier_sizes.append(frontier.qsize() + 1)  # Adjust to include the current node just popped

        if current == goal:
            path = []
            while current in reached:
                path.append(current)
                current = reached[current]["parent"]
            return path[::-1], expanded, frontier_sizes

        expanded.append(current)

        for successor in expand(grid, current):
            successor_cost = current_cost + cost(grid, successor)  # Assuming cost() computes the cost of moving to successor
            if successor not in reached or successor_cost < reached[successor]["cost"]:
                reached[successor] = {"cost": successor_cost, "parent": current}
                priority = successor_cost + heuristic_cost(heuristic, successor, goal)
                frontier.put((priority, successor))

        # If Beam Search, limit the frontier size
        if mode == PathPlanMode.BEAM_SEARCH:
            new_frontier = PriorityQueue()
            temp_list = []
            while not frontier.empty():
                temp_list.append(frontier.get())
            # Sort to ensure highest priorities are kept, assuming lower values are better
            temp_list.sort(key=lambda x: x[0])
            for item in temp_list[:width]:  # Ensure we only keep up to 'width' items
                new_frontier.put(item)
            frontier = new_frontier

    # Goal not reached
    return [], expanded, frontier_sizes

def ida_star(grid, start, goal, heuristic: Heuristic):
    """ Performs IDA* search to find the shortest path from
    start to goal in the gridworld.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.

    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    bound = 0
    frontier_sizes = []
    while True:
        path, expanded, frontier_size, new_bound = __dfs_ida_star(grid, start, goal, heuristic, bound)
        frontier_sizes += frontier_size

        if len(path) > 0 or np.isinf(new_bound):
            return path, expanded, frontier_sizes
        else:
            bound = new_bound


def __dfs_ida_star(grid, start, goal, heuristic, bound):
    frontier = [start]  # Initialize with the start node
    expanded = []  # List to keep track of expanded nodes
    frontier_sizes = []  # Record the size of the frontier over time
    reached = {start: {"cost": 0, "parent": None}}  # Maps nodes to their cost and parent
    next_bound = np.inf  # The next bound to use if the current bound is exceeded

    while frontier:
        node = frontier.pop()  # LIFO: Last In, First Out
        if node != goal:
            expanded.append(node)  # Add node to expanded list if it's not the goal

        # Directly handle the goal node check to avoid expanding it
        if node == goal:
            # Reconstruct the path from goal to start
            path = [node]
            while reached[node]["parent"] is not None:
                node = reached[node]["parent"]
                path.append(node)
            return path[::-1], expanded, frontier_sizes, next_bound

        frontier_sizes.append(len(frontier))  # Update the size of the frontier after popping

        for child in expand(grid, node):
            g = reached[node]["cost"] + cost(grid, child)  # Cumulative cost to reach the child
            if heuristic == Heuristic.EUCLIDEAN:
                h = numpy.sqrt((goal[0] - child[0])**2 + (goal[1] - child[1])**2)
            elif heuristic == Heuristic.MANHATTAN:
                h = abs(goal[0] - child[0]) + abs(goal[1] - child[1])
            total = g + h

            # Check if the child node should be added to the frontier
            if total <= bound:
                if child not in reached or reached[child]["cost"] > g:
                    frontier.append(child)
                    reached[child] = {"cost": g, "parent": node}
            else:
                # Update the next_bound if the total cost exceeds the current bound
                if total < next_bound:
                    next_bound = total

        # If the loop ends without finding a path, the goal was not reached within this bound
        if not frontier:  # Frontier is empty, no path found within the current bound
            break

    # Return an empty path if the goal is not reached, along with the expanded nodes and the next bound
    return [], expanded, frontier_sizes, next_bound



def test_world(world_id, start, goal, h, width, animate, world_dir):
    print(f"Testing world {world_id}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")

    if h == 1 or h == 2:
        modes = [
            PathPlanMode.A_STAR,
            PathPlanMode.BEAM_SEARCH
        ]
    elif h == 3 or h == 4:
        h -= 2
        modes = [
            PathPlanMode.IDA_STAR
        ]
    else:
        modes = [
            PathPlanMode.DFS,
            PathPlanMode.BFS
        ]

    for mode in modes:

        search_type, path, expanded, frontier_size = None, None, None, None
        if mode == PathPlanMode.DFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "DFS"
        elif mode == PathPlanMode.BFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "BFS"
        elif mode == PathPlanMode.A_STAR:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, 0)
            search_type = "A_STAR"
        elif mode == PathPlanMode.BEAM_SEARCH:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, width)
            search_type = "BEAM_A_STAR"
        elif mode == PathPlanMode.IDA_STAR:
            path, expanded, frontier_size = ida_star(grid, start, goal, h)
            search_type = "IDA_STAR"
        
        if search_type != None:
            path_cost = 0
            for c in path:
                path_cost += cost(grid, c)

            print(f"Mode: {search_type}")
            print(f"Path length: {len(path)}")
            print(f"Path cost: {path_cost}")
            print(f"Number of expanded states: {len(frontier_size)}")
            print(f"Max frontier size: {max(frontier_size) if len(frontier_size) > 0 else 0}\n")
            if animate == 0 or animate == 1:
                visualize_expanded(grid, start, goal, expanded, path, animation=animate)
            else:
                visualize_path(grid, start, goal, path)