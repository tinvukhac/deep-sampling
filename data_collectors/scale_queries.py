def main():
    print ("Scale query to world MBR")
    min_x, min_y, max_x, max_y = -180.0, -83.0, 180.0, 90.0
    input_f = open('../data/queries.csv')
    output_f = open('../data/scaled_queries.csv', 'w')

    line = input_f.readline()
    while line:
        data = line.strip().split(',')
        x = min_x + float(data[0]) * (max_x - min_x)
        y = min_y + float(data[1]) * (max_y - min_y)
        output_f.writelines('{},{}\n'.format(x, y))

        line = input_f.readline()

    output_f.close()
    input_f.close()


if __name__ == "__main__":
    main()
