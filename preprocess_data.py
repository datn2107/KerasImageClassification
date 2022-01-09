import os
import argparse
import pandas as pd


def get_class_map(class_list_fpath):
    class_map = {}
    class_names = []

    with open(class_list_fpath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, class_name = int(line.split()[0]), line.split()[1]
            class_map[class_id] = class_name
            class_names.append(class_name)

    return class_names, class_map


def create_new_df(df, new_columns, class_map):
    new_data = []

    for index, label in df.iterrows():
        class_id = label[0]

        new_row = {key: int(key == class_map[class_id]) for key in new_columns}
        new_row['filename'] = index

        new_data.append(new_row)

    new_df = pd.DataFrame(new_data).set_index('filename')
    return new_df


def is_similar_df(df, new_df, class_map):
    for index, label in df.iterrows():
        class_id = label[0]
        class_name = class_map[class_id]

        if new_df.loc[index, class_name] == 0:
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Data directory',
                        default='../data')
    parser_args = parser.parse_args()
    data_dir = parser_args.data_dir

    class_names, class_map = get_class_map(os.path.join(data_dir, 'class_list.txt'))
    new_columns = ['filename']
    new_columns = new_columns.extend(class_names)

    for phase in ['train', 'val', 'test']:
        df_path = os.path.join(data_dir, phase + '_labels.csv')
        if not os.path.exists(df_path):
            continue

        df = pd.read_csv(df_path, index_col=0)
        if len(df.columns) == 1:
            new_df = create_new_df(df, new_columns, class_map)
            if is_similar_df(df, new_df, class_map):
                new_df.to_csv(df_path)
            else:
                print('Cannot save new metadata. Something went wrong !!!')
