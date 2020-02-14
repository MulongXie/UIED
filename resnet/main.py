import Resnet
import Data

data = Data.Data()
data.load_data(max_number=1000)
data.generate_training_data(0)

resnet = Resnet.ResClassifier()
# resnet.train(data)
# root = 'E:/Mulong/Datasets/rico/elements-14-2/ToggleButton'
resnet.evaluate(data)
