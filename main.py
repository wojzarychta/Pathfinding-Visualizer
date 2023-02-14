import heapq as heap
import sys
from collections import defaultdict

import pygame

# CONSTANTS
DIS_WIDTH = 800  # width of window in px
DIS_HEIGHT = 800  # height of window in px

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (169, 169, 169)
LIGHT_BLUE = (73, 221, 235)
BLUE = (35, 179, 232)
GREEN = (52, 227, 110)
ORANGE = (255, 94, 5)

dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
pygame.display.set_caption('Pathfinding visualizer')


class Cell:
    """
    class modeling single cell in grid
    """
    def __init__(self, row, col, width=20, height=20):
        self.row = row
        self.col = col
        self.width = width  # width of cell in px
        self.height = height  # height of cell in px
        self.pos = (col * self.width, row * self.height)  # top left corner of cell
        self.color = WHITE
        self.visited = False
        self.neighbours = []  # adjacent cells

    def draw_cell(self):
        pygame.draw.rect(dis, self.color, (self.pos, (self.width, self.height)))

    def make_wall(self):
        self.color = BLACK

    def is_wall(self):
        return self.color == BLACK

    def set_start(self):
        self.color = LIGHT_BLUE

    def set_finish(self):
        self.color = BLUE

    def is_empty(self):
        return self.color == WHITE

    def set_visited(self):
        self.visited = True
        self.color = GREEN

    def set_active(self):
        self.color = (0, 255, 0)

    def set_path(self):
        self.color = ORANGE

    def reset(self):
        self.color = WHITE
        self.visited = False
        self.neighbours = []

    def clear(self):
        if not self.is_wall():
            self.color = WHITE
        self.visited = False
        self.neighbours = []

    def add_neighbours(self, grid, rows, cols):
        if self.row > 0 and not grid[self.row - 1][self.col].is_wall():  # upper neighbour
            self.neighbours.append(grid[self.row - 1][self.col])
        if self.col < cols - 1 and not grid[self.row][self.col + 1].is_wall():  # right neighbour
            self.neighbours.append(grid[self.row][self.col + 1])
        if self.row < rows - 1 and not grid[self.row + 1][self.col].is_wall():  # bottom neighbour
            self.neighbours.append(grid[self.row + 1][self.col])
        if self.col > 0 and not grid[self.row][self.col - 1].is_wall():  # left neighbour
            self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False  # added to make pq work


class Grid:
    def __init__(self, width, height, rows, cols):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.cell_width = width // cols
        self.cell_height = height // rows
        self.grid = self.generate_grid()

    def generate_grid(self):
        # generates grid as list[list[Cell]]
        grid = []
        for row in range(self.rows):
            grid.append([])
            for col in range(self.cols):
                cell = Cell(row, col, self.cell_width, self.cell_height)
                grid[row].append(cell)
        return grid

    def update_graph(self):
        # update neighbours of every cell
        for row in self.grid:
            for cell in row:
                cell.add_neighbours(self.grid, self.rows, self.cols)

    def reset(self):
        # resets whole grid to default state
        for row in self.grid:
            for cell in row:
                cell.reset()

    def clear(self):
        # clears grid without wiping walls
        for row in self.grid:
            for cell in row:
                cell.clear()

    def draw_grid(self):
        # draws grid
        dis.fill(BLACK)
        for row in self.grid:
            for cell in row:
                cell.draw_cell()
        self.__draw_lines()

    def __draw_lines(self):
        # helper function
        for i in range(1, self.rows):
            pygame.draw.line(dis, GREY, (0, i * self.cell_height), (self.height, i * self.cell_height))
        for i in range(1, self.cols):
            pygame.draw.line(dis, GREY, (i * self.cell_width, 0), (i * self.cell_width, self.width))


def get_clicked_cell(grid: Grid):
    # returns clicked row and cell of grid as tuple
    pos = pygame.mouse.get_pos()
    col = pos[0] // grid.cell_width
    row = pos[1] // grid.cell_height
    return row, col


def draw():
    grid.draw_grid()
    pygame.display.update()


def DFS(v, target):
    if v == target:  # goal
        draw_path(start, target)
        return True
    v.set_visited()  # add to visited cells
    draw()
    for neighbour in v.neighbours:
        # recursively visit all neighbours
        if not neighbour.visited:
            path[neighbour] = v  # track path from source
            if DFS(neighbour, target):
                return True
    return False


def BFS(source, target):
    queue = [source]
    visited = set()
    visited.add(source)
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        v = queue.pop(0)
        if v == target:
            draw_path(source, target)
            return True

        for neighbour in v.neighbours:
            # visit all the neighbours of v and enqueue them
            if neighbour not in visited:
                visited.add(neighbour)
                path[neighbour] = v  # track path from source
                queue.append(neighbour)
                neighbour.set_active()

        draw()

        if v != source:
            v.set_visited()

    return False


def dijkstra(source, target):
    visited = set()
    pq = []
    node_costs = defaultdict(lambda: int(sys.maxsize))  # initialize cost of every new node to inf
    node_costs[source] = 0
    heap.heappush(pq, (0, source))

    while pq:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()

        _, v = heap.heappop(pq)  # get vertex with minimum cost
        visited.add(v)

        if v == target:  # goal
            draw_path(source, target)
            return True

        for neighbour in v.neighbours:
            if neighbour in visited:  # if already visited by algorithm
                continue

            new_cost = node_costs[v] + 1  # as it is unweighted graph new cost is always incremented
            if node_costs[neighbour] > new_cost:  # the current distance is greater than new distance
                path[neighbour] = v  # track path from source
                node_costs[neighbour] = new_cost  # update cost
                heap.heappush(pq, (new_cost, neighbour))  # push to priority queue with new cost
                neighbour.set_active()

            if v != source:
                v.set_visited()

            draw()

    return False


def heuristics(current: Cell, goal: Cell):
    # manhattan heuristics for A* algorithm
    return abs(current.row - goal.row) + abs(current.col - goal.col)


def a_star(source, target):
    # additional variable to keep track of order in which vertices were put on pq as pq is not FIFO
    count = 0
    open_set = []
    closed_set = {source}

    heap.heappush(open_set, (0, count, source))

    # initialize cost of every new node to inf
    g = defaultdict(lambda: int(sys.maxsize))  # cost to move from the starting point to a given square on the grid
    f = defaultdict(lambda: int(sys.maxsize))  # estimated cost to move from given square to the final destination
    g[source] = 0
    f[source] = heuristics(source, target)

    while open_set:
        _, _, current = heap.heappop(open_set)  # get vertex with minimum f cost
        closed_set.add(current)  # add to closed set (visited)

        if current == target:  # goal
            draw_path(source, target)
            return True

        for neighbour in current.neighbours:
            if neighbour in closed_set:  # if vertex on the closed list, ignore it
                continue

            temp_g = g[current] + 1  # as it is unweighted graph, g is always incremented

            # if not in open set or new g cost is lower (better path)
            if neighbour not in [entry[2] for entry in open_set] or temp_g < g[neighbour]:
                path[neighbour] = current  # keep track of path from source
                # update g and f cost
                g[neighbour] = temp_g
                f[neighbour] = temp_g + heuristics(neighbour, target)
                count += 1
                heap.heappush(open_set, (f[neighbour], count, neighbour))  # add to open set
                neighbour.set_active()

        draw()

        if current != source:
            current.set_visited()

    return False


def draw_path(source, target):
    # draw path traversing from vertex to its parent as long as source is reached
    curr = target  # start from target
    curr.set_path()
    while True:
        curr = path[curr]
        curr.set_path()
        grid.draw_grid()
        pygame.display.update()
        if curr == source:
            return


if __name__ == '__main__':
    grid = Grid(DIS_WIDTH, DIS_HEIGHT, 40, 40)
    start = None
    finish = None
    path = {}
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if pygame.mouse.get_pressed()[0]:  # left mouse button
                row, col = get_clicked_cell(grid)
                cell = grid.grid[row][col]
                if cell.is_empty():
                    # firstly draw start and finish, then walls
                    if not start and cell != finish:
                        start = cell
                        start.set_start()
                    elif not finish and cell != start:
                        finish = cell
                        finish.set_finish()
                    else:
                        cell.make_wall()
            elif pygame.mouse.get_pressed()[2]:  # right mouse button
                row, col = get_clicked_cell(grid)
                cell = grid.grid[row][col]
                if not cell.is_empty():
                    # erase cell
                    if cell == start:
                        start = None
                    elif cell == finish:
                        finish = None
                    cell.reset()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_c:  # clear board
                    grid.clear()
                    start.set_start()
                    finish.set_finish()
                    path = {}

                elif event.key == pygame.K_r:  # reset board without wiping walls
                    start = None
                    finish = None
                    grid.reset()
                    path = {}

                if start is not None and finish is not None:  # start only if start and finish are set
                    grid.update_graph()

                    if event.key == pygame.K_d:  # DFS algorithm
                        DFS(start, finish)

                    elif event.key == pygame.K_b:  # BFS algorithm
                        BFS(start, finish)

                    elif event.key == pygame.K_s:  # Dijkstra's algorithm
                        dijkstra(start, finish)

                    elif event.key == pygame.K_a:  # A* algorithm
                        a_star(start, finish)

        draw()
