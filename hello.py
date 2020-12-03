import paddlehub as hub
import paddlehub
paddlehub.server_check()

# module = hub.Module(name="ace2p")
# res = module.segmentation(paths= ["./test_image.jpg"], visualization=True, output_dir='ace2p_output')

# module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
# res = module.face_detection(
#     paths=["./test_image.jpg"], visualization=True, output_dir='face_detection_output')

# module = hub.Module(name="human_pose_estimation_resnet50_mpii")
# res = module.keypoint_detection(
#     paths=["./test_image.jpg"], visualization=True, output_dir='keypoint_output')

# lac = hub.Module(name="lac")
# test_text = [
#     "1996年，曾经是微软员工的加布·纽维尔和麦克·哈灵顿一同创建了Valve软件公司。他们在1996年下半年从id software取得了雷神之锤引擎的使用许可，用来开发半条命系列。"]
# res = lac.lexical_analysis(texts=test_text)
# print("The resuls are: ", res)

senta = hub.Module(name="senta_bilstm")
test_text = ["味道不错，确实不算太辣，适合不能吃辣的人。就在长江边上，抬头就能看到长江的风景。鸭肠、黄鳝都比较新鲜。"]
res = senta.sentiment_classify(texts=test_text)
print(res)

print("hello")
