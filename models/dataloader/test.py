# 写一个读取csv文件的函数
def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines][1:]
    return lines

# 写一个统计输入标签有多少类的函数
def count_class(csv_path):
    lines = read_csv(csv_path)
    label = []
    for l in lines:
        context = l.split(',')
        wnid = context[1]
        if wnid not in label:
            label.append(wnid)
    return len(label)

if __name__ == '__main__':
    x = count_class('/home/lupeiyu/datasets/RENet/{}/split/train.csv'.format('miniimagenet'))
    print(x)
    x = count_class('/home/lupeiyu/datasets/RENet/{}/split/train.csv'.format('cub'))
    print(x)
    x = count_class('/home/lupeiyu/datasets/RENet/cars/train.csv')
    print(x)
    x = count_class('/home/lupeiyu/datasets/RENet/dogs/For_FewShot/train.csv')
    print(x)
    x = count_class('/home/lupeiyu/datasets/RENet/flowers/train.csv')
    print(x)