import numpy as np
import pygame
import csv

pygame.init()
WIDTH, HEIGHT = 900, 900
LR = .1
RADIOS = 40
MAX_ATTEMPETS = 10
COLORS = ["darkmagenta", "gray26", "blue4", "chartreuse3", "darkturquoise", "cornflowerblue", "darkgreen", "darkorange",
          "blue", "darkolivegreen2", "deepskyblue3", "firebrick1", "gold2"]
PARTYS = ["Labour", "Yamina", "Yahadot Hatora", "The Joint Party", "Zionut Datit", "Kachul Lavan", "Israel Betinu",
          "Licod", "Merez", "Raam", "Yesh Atid", "Shas", "Tikva Hadasha"]
MINIMUM = (float('inf'), 0)
FINISHED = False
TO_PRINT = False
AFTER_FRAME = 0
FONT = pygame.font.SysFont('comicsans', 30)


class Hexagon:
    """
    this class represent our hexagon.
    """

    def __init__(self, center, pos, id, order):
        """
        init the hexagon
        :param center: the center to draw the hexagon
        :param pos: the pos in space
        :param id: its number to calculate faste distances
        :param order: in order to acount the suffled values in featurs
        """
        self.colors = [COLORS[index] for index in order]
        self.distances = None
        self.pos = pos
        self.idx = id
        self.value = create_rand_value()
        self.center = center
        self.vertices = create_vertices(self.center)
        self.features = []

    def draw(self, screen):
        """
        this function draw the hexagon on the screen
        :param screen: our screen to draw on
        :return:
        """
        # get which party got most votes and use it's color
        color = self.colors[np.argmax(self.value[1:])]
        pygame.draw.polygon(screen, color, self.vertices)

    def update(self, closes, loss):
        """
        this function update the vector of the hexagon based on the equation  V(t+1)=V(t)+ LR * H(x) * loss
        :param closes: the hexagon closes to the feture
        :param loss: the loss value
        :return:
        """
        # the distance between this hexagon and the closes hexagon
        n = 1 if closes is self else (distance_between_points(self.pos, closes.pos))
        self.value += LR * (1 / n ** 2) * loss

    def loss(self, vector):
        """
        this function calculate the loss value
        :param vector: the vector of the feature
        :return: the loss value
        """
        return vector - self.value

    def distanceToFeature(self, vector):
        """
        this function calculate the distance in space between the feature vector and the hexagon vector
        :param vector: the feature vector
        :return: the distance
        """
        return np.linalg.norm(self.value - vector)

    def print(self):
        """
        this function print all the citys inside the hexagon
        :return:
        """
        to_print = f"Hex id {self.idx} got -> "
        for f in self.features:
            to_print += f.name + ", "
        if len(self.features) == 0:
            to_print += "Nothing  "
        print(to_print[:-2])

    def lossTotal(self):
        """"
        calculate the loss of the hexagon based on the sum of the distance to each feature inside the hexagon
        """
        losses = [self.distanceToFeature(f.vector) for f in self.features]
        return sum(losses)


def create_rand_value():
    """
    creat random vector for the hexagon
    :return:
    """
    return np.asarray(np.zeros(14), dtype = float)


class Feature:
    """
    this class for the feature, each city is it on feature
    """

    def __init__(self, name, vector):
        """
        init the feature
        :param name:the name of the feature
        :param vector:the vector of that featur
        """
        self.name = name
        self.vector = vector
        self.hex = -1

    def step(self, cells):
        """
        this function calculate the steps to update the hexagon
        :param cells: hexagons to update
        :return:
        """
        # calculate the distance for all the hexagons to this feature vector
        dis_len = [(cell.distanceToFeature(self.vector), cell) for cell in cells]
        # get the clossest hexagon
        minimum = min(dis_len, key = lambda k: k[0])
        self.hex = minimum[1]
        loss = self.hex.loss(self.vector)
        self.hex.features.append(self)
        for hex in cells:
            # update all hexagons
            hex.update(self.hex, loss)


def create_vertices(position, vertex_count=6, radius=RADIOS):
    """
    this function creat all the vertices
    :param position: the center of the polygon
    :param vertex_count: how many vertices to create
    :param radius: what distance to put each vertice
    :return:the vertices list
    """
    n, r = vertex_count, radius
    x, y = position
    return [(x - r * np.cos(np.pi / 2 + 2 * np.pi * i / n),
             y - r * np.sin(np.pi / 2 + 2 * np.pi * i / n)) for i in range(n)]


def distance_between_points(p1, p2):
    """
    calculating distance between two points in space
    :param p1: point 1
    :param p2: point 2
    :return: the distance
    """
    point1 = np.asarray(p1)
    point2 = np.asarray(p2)
    return np.linalg.norm(point1 - point2)


def create_features(data, suffled_order):
    """
    create all the the features
    :param data: the data to create the features
    :param suffled_order: the suffled_order that we randomize the feature order in the vector
    :return: list of all the features
    """
    features = []
    for row in data:
        name = row[0]
        vector = [(row[1])]
        vector.extend(row[index + 3] for index in suffled_order)
        features.append(Feature(name, np.asarray(vector, dtype = float)))
    return features


def update(hexes, features):
    """
    this function updates each iteartion all the features
    :param hexes: the hexagons list
    :param features: the features list
    :return:
    """
    global MINIMUM
    global FINISHED
    global TO_PRINT
    mini_loss, times = MINIMUM
    # while the map not finished
    if not FINISHED:
        for h in hexes:
            h.features = []
        # suffle all the features order
        np.random.shuffle(features)
        for f in features:
            f.step(hexes)
        # calculate the loss of the map
        losss = [h.lossTotal() for h in hexes]
        total_loss = sum(losss)
        # check if it's the lowest loss
        if total_loss < mini_loss:
            mini_loss = total_loss
            times = 1
        else:
            times += 1
        MINIMUM = (mini_loss, times)
        # if the map didn't lower in loss in the last 5 iteartion
        if times > 5:
            FINISHED = True


def init(data, hexes_pos):
    """
    this function init each attampet of the map
    :param data:  the data from the file
    :param hexes_pos: the list of all the hexagons position
    :return: list of features and list of the hexagons
    """
    suffled_order = [index for index in range(len(PARTYS))]
    np.random.shuffle(suffled_order)
    features = create_features(data, suffled_order)
    hexes = [Hexagon(center, pos, i, suffled_order) for i, (center, pos) in enumerate(hexes_pos)]
    return features, hexes


if __name__ == "__main__":

    file = input("Please enter a path to file\n")
    #read the file
    data = np.loadtxt(file, skiprows = 1, delimiter = ",", dtype = str)
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    FPS = 60
    generation = 1
    party_texts = []
    t_x = 50
    SOM_map = []
    partys_in_row = 4
    #create all the partys texts
    for index, party in enumerate(PARTYS):
        i = index // partys_in_row
        j = index % partys_in_row
        if j == 0:
            t_x = 50
        y = 670 + i * 50
        x = t_x
        text_box = FONT.render(party, True, COLORS[index])
        pos = [x, y]
        party_texts.append((text_box, pos))
        t_x += text_box.get_width() + 25
    hexes_pos = []
    start_x = 150
    start_y = 100
    #calculate the position of the all hexagons
    for i in range(9):
        num_in_row = 9 - abs(4 - i)
        for j in range(num_in_row):
            add_x = RADIOS * (abs(4 - i))
            x = start_x + add_x + j * 2 * RADIOS
            y = start_y + 1.5 * i * RADIOS
            hexes_pos.append(([x, y], [i, j]))
    features, hexes = init(data, hexes_pos)
    while True:
        clock.tick(FPS)
        pygame.draw.rect(win, 'white', [0, 0, WIDTH, HEIGHT])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        #update the features and hexagons
        update(hexes, features)
        #if the map finished calculating
        if FINISHED and AFTER_FRAME > 1 * FPS and generation <= MAX_ATTEMPETS:
            SOM_map.append((features, hexes, MINIMUM[0]))
            FINISHED = False
            AFTER_FRAME = 0
            generation += 1
            if generation <= MAX_ATTEMPETS:
                #get all new features and hexagons
                features, hexes = init(data, hexes_pos)
                MINIMUM = (float('inf'), 0)
            TO_PRINT = False if generation <= 10 else True
        if TO_PRINT:
            FINISHED = True
            #get the lowest loss map
            features, hexes, mini_loss=min(SOM_map,key = lambda x: x[2])
            #printing all the mapping
            for hex in hexes:
                hex.print()
            TO_PRINT = False
        genText = FONT.render(f"Attempt :{generation}", True, "blue")
        win.blit(genText, [300, 20])
        for hex in hexes:
            hex.draw(win)
        for text, pos in party_texts:
            win.blit(text, pos)
        if FINISHED:
            AFTER_FRAME += 1
        pygame.display.flip()
