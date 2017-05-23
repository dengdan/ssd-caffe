import util

image_dir = "/home/dengdan/dataset_nfs/ICDAR2015/Challenge4/ch4_training_images"
test_result = "/home/dengdan/temp_nfs/results/icdar/SSD_600x600/comp4_det_test_text.txt"
dump_path = "~/temp_nfs/result/ssd600x600_54000/"
lines = util.io.read_lines(test_result);
result_dict = {}

SHOW_NUM = 10
for line in lines:
    data = line.split()
    image_name, score, x, y, w, h = data[0], float(data[1]), int(data[2]), int(data[3]), int(data[4]), int(data[5])
    if image_name not in result_dict:  
        result_dict[image_name] = []
    result_dict[image_name].append([score, x, y, w, h])
    
for image_name in result_dict:
    image_path = util.io.join_path(image_dir, image_name + '.jpg');
    image = util.img.imread(image_path)
    bboxes = result_dict[image_name];
    num = len(bboxes)
    if num > SHOW_NUM:
        num = SHOW_NUM
        
    for i in range(num):
        score, x1, y1, x2, y2 = bboxes[i]
        util.img.rectangle(image, (x1, y1), (x2, y2), color = util.img.COLOR_GREEN, border_width = 2)
        util.img.put_text(image, text = str(score), pos = (x1, y1), scale = 0.5, color = util.img.COLOR_WHITE, thickness = 1);
        util.img.imwrite(util.io.join_path(dump_path, image_name + '.jpg'), image)
    

