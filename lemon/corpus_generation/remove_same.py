import sys
import hashlib
import random
# random.seed(0)
def gen_md5(data):
    """
        生成md5
    :param data: 字符串数据
    :return:
    """
    md5 = hashlib.md5()
    md5.update(data.encode('utf-8'))
    return md5.hexdigest()
    
def big_file_remove_same(input_file, output_file):
    """
        针对大文件文件去重（将文件文件写在一行的，没有办法去重）
    :param input_file:
    :param output_file:
    :return:
    """
    finger_print_set = set()
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as ff,open('/tmp/dumple.txt', 'w', encoding='utf-8') as fff:
        f = f.readlines()
        # random.shuffle(f)
        for line in f:
            line_string = line.strip()
            finger_print = gen_md5(line_string)
            if finger_print not in finger_print_set:
                finger_print_set.add(finger_print)
                ff.write(line)
            else:
                fff.write(line)

    
if __name__ == "__main__":
    # big_file_remove_same(sys.argv[1],sys.argv[2])
    input_file = './table-evidence.txt'
    output_file = './result.txt'
    big_file_remove_same(input_file, output_file)