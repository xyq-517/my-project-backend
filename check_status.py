import os

# Check prediction results count
pred_dir = "miou_out/detection-results"
pred_count = len([f for f in os.listdir(pred_dir) if f.endswith('.png')])
print("Prediction results:", pred_count)

# Check test.txt count
test_txt = "VOCdevkit_kits19/VOC2007/ImageSets/Segmentation/test.txt"
test_count = len(open(test_txt).read().splitlines())
print("Test.txt images:", test_count)

# Check if they match
if pred_count == test_count:
    print("OK: All test images have predictions!")
else:
    print("Missing predictions:", test_count - pred_count)