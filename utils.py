import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from enum import IntEnum
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from typing import Optional, List, Tuple
from pathlib import Path


class Environment(IntEnum):
    FLATLAND = 0
    POND = 1
    VALLEY = 2
    MOUNTAIN = 3
    EXPANDED = 4


class PathPlanMode(IntEnum):
    DFS = 1
    BFS = 2
    A_STAR = 3
    BEAM_SEARCH = 4
    IDA_STAR = 5


class Heuristic(IntEnum):
    MANHATTAN = 1
    EUCLIDEAN = 2


def cost(
    grid: npt.ArrayLike, 
    point: Tuple[int, int],
) -> int:
    if grid[point] == Environment.FLATLAND:
        return 3
    elif grid[point] == Environment.POND:
        return 2
    elif grid[point] == Environment.VALLEY:
        return 5
    elif grid[point] == Environment.MOUNTAIN:
        return np.inf


def expand(
    grid: npt.ArrayLike, 
    point: Tuple[int, int],
) -> List[Tuple[int, int]]:
    children = []
    neighbors = [[0, 1], [1, 0], [0, -1], [-1, 0],
                 [1, 1], [1, -1], [-1, 1], [-1, -1]]
    
    x, y = point
    for i, j in neighbors:
        if x + i >= 0 and x + i < grid.shape[0] \
        and y + j >= 0 and y + j < grid.shape[1] \
        and grid[x + i, y + j] != Environment.MOUNTAIN:
            children.append((x + i, y + j))
    return children


def create_pond(
    grid: npt.ArrayLike,
    center_x: int,
    center_y: int,
    axis_x: int,
    axis_y: int,
) -> npt.ArrayLike:
    c_x, c_y, a, b = center_x, center_y, axis_x, axis_y
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if ((x - c_x) / a) ** 2 + ((y - c_y) / b) ** 2 < 1:
                grid[int(x), int(y)] = Environment.POND
    return grid


def create_valley(
    grid: npt.ArrayLike, 
    center_x: int, 
    center_y: int, 
    radius: int,
) -> npt.ArrayLike:
    c_x, c_y, r_2 = center_x, center_y, radius
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2) < r_2:
                grid[int(x), int(y)] = Environment.VALLEY
    return grid


def create_mountain(
    grid: npt.ArrayLike,
    lower_x: int,
    upper_x: int,
    lower_y: int,
    upper_y: int,
) -> npt.ArrayLike:
    grid[lower_x:upper_x, lower_y:upper_y] = Environment.MOUNTAIN
    return grid


def highlight_start_and_end(
    grid: npt.ArrayLike,
    cell: Tuple[int, int],
    val: int,
) -> npt.ArrayLike:
    c_x, c_y = cell
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2) < 5:
                grid[int(x), int(y)] = val

    return grid


def sample_world_1(
    width: int = 100, 
    height: int = 100,
) -> npt.ArrayLike:
    start = (10, 10)
    goal = (87, 87)
    grid_world = np.zeros((width, height))
    
    grid_world = create_pond(grid_world, 87, 40, 12, 12)

    grid_world = create_valley(grid_world, 40, 87, 12)

    grid_world = create_mountain(grid_world, 0, 35, 45, 50)
    grid_world = create_mountain(grid_world, 45, 50, 0, 35)
    grid_world = create_mountain(grid_world, 15, 75, 70, 75)
    grid_world = create_mountain(grid_world, 70, 75, 15, 75)
    
    return grid_world, start, goal


def sample_world_2(
    width: int = 100,
    height: int = 100,
) -> npt.ArrayLike:
    start = (10, 10)
    goal = (90, 90)
    grid_world = np.zeros((width, height))
    
    grid_world = create_pond(grid_world, 37, 10, 20, 7)
    grid_world = create_pond(grid_world, 49, 22, 7, 7)
    grid_world = create_pond(grid_world, 50, 22, 7, 7)
    grid_world = create_pond(grid_world, 49, 78, 7, 7)
    grid_world = create_pond(grid_world, 50, 78, 7, 7)
    grid_world = create_pond(grid_world, 63, 90, 20, 7)
    grid_world = create_pond(grid_world, 20, 50, 10, 10)
    grid_world = create_pond(grid_world, 80, 50, 10, 10)

    grid_world = create_valley(grid_world, 50, 50, 20)

    grid_world = create_mountain(grid_world, 20, 40, 20, 25)
    grid_world = create_mountain(grid_world, 60, 100, 20, 25)
    grid_world = create_mountain(grid_world, 0, 40, 75, 80)
    grid_world = create_mountain(grid_world, 60, 80, 75, 80)
    return grid_world, start, goal


def sample_world_3(
    width: int = 100,
    height: int = 100,
) -> npt.ArrayLike:
    start = (10, 10)
    goal = (90, 90)
    grid_world = np.zeros((width, height))

    grid_world = create_pond(grid_world, 25, 10, 7, 7)
    grid_world = create_pond(grid_world, 30, 10, 7, 7)
    grid_world = create_pond(grid_world, 49, 10, 30, 7)
    grid_world = create_pond(grid_world, 66, 10, 30, 7)
    grid_world = create_pond(grid_world, 85, 10, 7, 7)
    grid_world = create_pond(grid_world, 90, 10, 7, 7)
    grid_world = create_pond(grid_world, 90, 15, 7, 7)
    grid_world = create_pond(grid_world, 90, 34, 7, 30)
    grid_world = create_pond(grid_world, 90, 50, 7, 7)
    grid_world = create_pond(grid_world, 90, 55, 7, 7)
    grid_world = create_pond(grid_world, 90, 60, 7, 7)
    grid_world = create_pond(grid_world, 90, 65, 7, 7)

    grid_world = create_valley(grid_world, 15, 40, 7)
    grid_world = create_valley(grid_world, 15, 60, 7)
    grid_world = create_valley(grid_world, 15, 80, 7)
    grid_world = create_valley(grid_world, 30, 50, 7)
    grid_world = create_valley(grid_world, 30, 70, 7)
    grid_world = create_valley(grid_world, 30, 90, 7)
    grid_world = create_valley(grid_world, 45, 40, 7)
    grid_world = create_valley(grid_world, 45, 60, 7)
    grid_world = create_valley(grid_world, 45, 80, 7)
    grid_world = create_valley(grid_world, 60, 50, 7)
    grid_world = create_valley(grid_world, 60, 70, 7)
    grid_world = create_valley(grid_world, 60, 90, 7)
    

    grid_world = create_mountain(grid_world, 10, 80, 20, 25)
    grid_world = create_mountain(grid_world, 75, 80, 20, 80)
    grid_world = create_mountain(grid_world, 80, 100, 75, 80)
    

    return grid_world, start, goal

def sample_world_4(
    width: int = 50,
    height: int = 50,
) -> npt.ArrayLike:
    start = (24, 24)
    goal = (43, 42)
    grid_world = np.zeros((width, height))
    
    grid_world = create_pond(grid_world, 7, 42, 7, 7)
    grid_world = create_pond(grid_world, 40, 24, 8, 5)
    grid_world = create_pond(grid_world, 24, 7, 6, 6)
    grid_world = create_pond(grid_world, 7, 24, 7, 5)

    grid_world = create_valley(grid_world, 24, 46, 4)
    grid_world = create_valley(grid_world, 32, 38, 4)
    grid_world = create_valley(grid_world, 7, 7, 6)
    grid_world = create_valley(grid_world, 42, 7, 6)
    
    grid_world = create_mountain(grid_world, 12, 20, 14, 19)
    grid_world = create_mountain(grid_world, 29, 38, 14, 19)
    grid_world = create_mountain(grid_world, 0, 15, 30, 35)
    grid_world = create_mountain(grid_world, 20, 50, 30, 35)

    return grid_world, start, goal


def visualize_grid_world(
        grid: npt.ArrayLike,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None,
) -> None:

    _, ax = plt.subplots()
    grid_world = np.copy(grid)

    cmap = ListedColormap([
        "#006600",  # Flatland
        "#4d94ff",  # Pond
        "#FFA500",  # Valley
        "#333333",  # Mountain
        "#00AA00",  # Start & Goal
    ])

    if start is not None:
        grid_world = highlight_start_and_end(grid_world, start, len(cmap.colors) - 1)
    if goal is not None:
        grid_world = highlight_start_and_end(grid_world, goal, len(cmap.colors) - 1)

    ax.imshow(grid_world, cmap=cmap)
    legend_elements = [
        Patch(facecolor="#006600", label="Flatland"),
        Patch(facecolor="#4d94ff", label="Pond"),
        Patch(facecolor="#FFA500", label="Valley"),
        Patch(facecolor="#333333", label="Mountain"),
        ]
    
    ax.set_title(f"Grid World Visualization")
    ax.legend(handles=legend_elements, 
              loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()


def visualize_path(
        grid: npt.ArrayLike,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        path: Path,
        blit: bool = False,
) -> None:
    
    fig, ax = plt.subplots()
    grid_world = np.copy(grid)

    cmap = ListedColormap([
        "#006600",  # Flatland
        "#4d94ff",  # Pond
        "#FFA500",  # Valley
        "#333333",  # Mountain
        "#00AA00",  # Start & Goal
    ])

    grid_world = highlight_start_and_end(grid_world, start, len(cmap.colors) - 1)
    grid_world = highlight_start_and_end(grid_world, goal, len(cmap.colors) - 1)

    ax.imshow(grid_world, cmap=cmap)

    legend_elements = [
        Patch(facecolor="#006600", label="Flatland"),
        Patch(facecolor="#4d94ff", label="Pond"),
        Patch(facecolor="#FFA500", label="Valley"),
        Patch(facecolor="#333333", label="Mountain"),
        ]

    path_line, = ax.plot([], [], color='#FF0000', label='Path')

    def update_path(frame):
        if frame < len(path):
            x, y = zip(*path[:frame+1])
            path_line.set_data(y, x)
        return path_line,

    _ = FuncAnimation(
        fig, 
        update_path, 
        frames=len(path), 
        repeat=False, 
        interval=1, 
        blit=blit
        )
    legend_elements.append(Patch(facecolor='#FF0000', label='Path'))

    ax.set_title(f"Grid World Path Planning Result")
    ax.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0), 
        ncol=5
        )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()


def visualize_expanded(
        grid: npt.ArrayLike,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        expanded: List[Tuple[int, int]],
        path: Optional[Path],
        animation: bool = True,
) -> None:
    
    fig, ax = plt.subplots()
    grid_world = np.copy(grid)

    cmap = ListedColormap([
        "#006600",  # Flatland
        "#4d94ff",  # Pond
        "#FFA500",  # Valley
        "#333333",  # Mountain
        "#86592d",  # Expanded
        "#00AA00",  # Start & Goal
    ])

    grid_world = highlight_start_and_end(grid_world, start, len(cmap.colors) - 1)
    grid_world = highlight_start_and_end(grid_world, goal, len(cmap.colors) - 1)

    legend_elements = []

    if path:
        path_x, path_y = zip(*path)
        gw, = ax.plot(path_y, path_x, color='#FF0000', label='Path')
        legend_elements.append(Patch(facecolor='#FF0000', label='Path'))

    # dumb bug fix
    fix_bug = grid_world[0, -1]
    grid_world[0, 3] = 4
    gw = ax.imshow(grid_world, cmap=cmap)
    grid_world[0, 3] = fix_bug

    legend_elements.extend([
        Patch(facecolor="#006600", label="Flatland"),
        Patch(facecolor="#4d94ff", label="Pond"),
        Patch(facecolor="#FFA500", label="Valley"),
        Patch(facecolor="#333333", label="Mountain"),
        Patch(facecolor="#86592d", label="Expanded"),
        ])

    expanded = [s for s in expanded if len(s) > 0]
    all_x, all_y = [], []

    if animation:
        def update_expanded(frame):
            if frame < len(expanded):
                expanded_grid_world = np.copy(grid_world)
                x, y = expanded[frame]
                all_x.append(x)
                all_y.append(y)
                expanded_grid_world[all_x, all_y] = Environment.EXPANDED
                gw.set_array(expanded_grid_world)
            return [gw]

        _ = FuncAnimation(
            fig, 
            update_expanded, 
            frames=len(expanded), 
            repeat=False, 
            interval=1, 
            )
    else:
        for s in expanded:
            x, y = s
            all_x.append(x)
            all_y.append(y)
        grid_world[all_x, all_y] = Environment.EXPANDED
        gw.set_array(grid_world)

    ax.set_title(f"Grid World Expanded Cells Result")
    ax.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0), 
        ncol=3
        )
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
