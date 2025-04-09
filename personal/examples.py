class ConstructionGame:

    def __init__(self, length, width):
        """
        :param length: (int) Length of the base
        :param width: (int) Width of the base
        """
        self.length = length
        self.width = width
        self.current = [[0 for _ in range(self.width)] for _ in range(self.length)]

    def add_cubes(self, cubes):
        """
        :param cubes: (list(bool)) The position of each cube to be dropped on the table
        """
        # so, you're always dropping length * width cubes when calling add_cubes
        # True values stack, False values don't stack
        # operating at just one pair level for current...
        for i in range(len(self.current)):
            for j in range(len(self.current[i])):
                if cubes[i][j] == True:
                    self.current[i][j] += 1
                    print('setting current at',i,j,'to + 1')
                elif cubes[i][j] == False:
                    self.current[i][j] = self.current[i][j] - 1 if self.current[i][j] > 0 else 0
                print(cubes[i][j])
                print(self.current)
                # print(cubes[i][j])
                # print(self.current[i][j])


    def height(self):
        """
        :returns: (int) The maximum vertical height in cubes
        """
        max_height = 0
        for row in self.current:
            max_height = max(max_height, max(row))
        return max_height

if __name__ == "__main__":
    game = ConstructionGame(2, 2)

    game.add_cubes([
        [True, True],
        [False, False]
    ])
    game.add_cubes([
        [True, True],
        [False, True]
    ])
    print(game.height())  # should print 2

    game.add_cubes([
        [False, False],
        [True, True]
    ])
    print(game.height())  # should print 1



# ls = [1,2,3]
# ls.extend((4,5,6))
# print(ls)

# l = r = 10
# for r in range(5):
#     print(r)
#     print(l)


# di = {}

# di['a'] = 1
# di['b'] = 2
# di['c'] = 3

# for k in di:
#     print(di[k])

# l = [1,2,3]
# print(sum(l[0:1]))

# print(3 ^ 2)

# a = 'abcdefg'~
# print(a[::-1])

# print(-float('inf')/2)

# print(type(str))

# if 10 > (20 or 9):
#     print(10)

# print('BY EOW SET UP TWO PROCEDURES, AN AD-HOC THAT JUST APPENDS TO A TABLE EVERYTHING FROM A CSV FILE (SINCE ITS JUST NEW DATA) SO APPEND ALL ROWS INTO EXISTING TABLE. SECOND PROCESS THAT DOES NEED TO CHECK FOR UPDATES, EITHER INSERT, UPDATE, OR DELETE VALUES'.capitalize())


# prices = []
# print(prices[0])


# for i in range(4,-1,-1):
#     print(i)


# print(
#     [1,2,3,4] * 3
# )


# #
# d = {
#     0: '1',
#     1: '2'
# }
# print(d)

# def count_up_to(max_value):
#     current = 1
#     while current <= max_value:
#         yield current
#         current += 1

# Using the generator
# print(num) for num in count_up_to(7)
# for num in count_up_to(5):
#     print(num)
