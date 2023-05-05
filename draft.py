def handle_input():
    r, c = list(map(int, input().strip('\n').split(' ')))
    m = int(input().strip('\n'))
    points = []
    for i in range(m):
        points.append(list(map(int, input().strip('\n').split(' '))))
    n = int(input().strip('\n'))
    stores = []
    for i in range(n):
        stores.append(list(map(int, input().strip('\n').split(' '))))

    return r, c, m, points, n, stores


def find_min_dis(point, stores):
    min_dis = 2147483647
    for store in stores:
        dis = get_dis(point, store)
        if dis < min_dis:
            min_dis = dis
    return min_dis



def get_dis(point, store):
    return abs(point[0] - store[0]) + abs(point[1] - store[1])

