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

dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
pygame.display.set_caption('Pathfinding visualizer')


class Cell:
    def __init__(self, row, col, width=20, height=20):
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.pos = (col * self.width, row * self.height)
        self.color = WHITE
        self.neighbours = []

    def draw_cell(self):
        pygame.draw.rect(dis, self.color, (self.pos, (self.width, self.height)))

    def make_wall(self):
        self.color = BLACK

    def is_wall(self):
        return  self.color == BLACK

    def set_start(self):
        self.color = LIGHT_BLUE

    def set_finish(self):
        self.color = BLUE

    def is_empty(self):
        return self.color == WHITE

    def reset(self):
        self.color = WHITE

    def add_neighbours(self, grid, rows, cols):
        if self.row > 0 and not grid[self.row - 1][self.col].is_wall():  # upper neighbour
            self.neighbours.append(grid[self.row - 1][self.col])
        if self.row < rows - 1 and not grid[self.row + 1][self.col].is_wall():  # bottom neighbour
            self.neighbours.append(grid[self.row - 1][self.col])
        if self.col > 0 and not grid[self.row][self.col - 1].is_wall():  # left neighbour
            self.neighbours.append(grid[self.row][self.col - 1])
        if self.col < cols - 1 and not grid[self.row][self.col + 1].is_wall():  # right neighbour
            self.neighbours.append(grid[self.row][self.col + 1])


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
        grid = []
        for row in range(self.rows):
            grid.append([])
            for col in range(self.cols):
                cell = Cell(row, col, self.cell_width, self.cell_height)
                grid[row].append(cell)
        return grid

    def update_graph(self):
        for row in self.grid:
            for cell in row:
                cell.add_neighbours(self.grid, self.rows, self.cols)

    def reset(self):
        for row in self.grid:
            for cell in row:
                cell.reset()

    def draw_grid(self):
        dis.fill(BLACK)
        for row in self.grid:
            for cell in row:
                cell.draw_cell()
        self.__draw_lines()

    def __draw_lines(self):
        for i in range(1, self.rows):
            pygame.draw.line(dis, GREY, (0, i * self.cell_height), (self.height, i * self.cell_height))
        for i in range(1, self.cols):
            pygame.draw.line(dis, GREY, (i * self.cell_width, 0), (i * self.cell_width, self.width))


def get_clicked_cell(grid: Grid):
    pos = pygame.mouse.get_pos()
    col = pos[0] // grid.cell_width
    row = pos[1] // grid.cell_height
    return row, col


if __name__ == '__main__':
    grid = Grid(DIS_WIDTH, DIS_HEIGHT, 40, 40)
    start = None
    finish = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if pygame.mouse.get_pressed()[0]:  # left mouse button
                row, col = get_clicked_cell(grid)
                cell = grid.grid[row][col]
                if cell.is_empty():
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
                    if cell == start:
                        start = None
                    elif cell == finish:
                        finish = None
                    cell.reset()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # start algorithm
                    pass
                elif event.key == pygame.K_c:
                    start = None
                    finish = None
                    grid.reset()

                elif event.key == pygame.K_d:  # DFS algorithm
                    grid.update_graph()

        grid.draw_grid()
        pygame.display.update()
