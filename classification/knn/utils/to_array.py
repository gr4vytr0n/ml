from numpy import zeros, float32

# turn data from tab seperated file into an array


def to_array(filename):
    with open(filename) as file:
        lines = []
        for file_line in file.readlines():
            lines.append(file_line)

        line_cnt = len(lines)
        item_cnt = len(lines[0].strip().split('\t'))
        features_array = zeros((line_cnt, (item_cnt - 1)), dtype=float32)
        labels_array = zeros((line_cnt), dtype=object)
        for i in range(len(lines)):
            line_list = lines[i].strip().split('\t')
            feature_list = line_list[0:(item_cnt - 1)]
            label_list = line_list[-1]
            features_array[i, :] = feature_list[0:]
            labels_array[i] = label_list

    return features_array, labels_array
