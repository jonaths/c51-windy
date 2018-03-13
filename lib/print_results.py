import matplotlib.pyplot as plt
import numpy as np

top_mavg_reward = [-0.72, -0.88, -0.08, -0.4, -0.72, -0.08, -0.56, -0.88, -1.04, -0.56, -0.72, -1.04, -0.56, 0.24, -0.56, 0.24, -0.24, 0.4, -0.72, 0.08, -0.72, -0.4, -0.56, -0.08, -0.08, -1.36, -0.88, -0.72, -1.04, -0.4, -0.56, 0.56, -1.36, -0.08, 0.24, -0.56, -1.04, -0.56, -0.08, -0.4, 0.24, -0.08, -0.4, -0.4, -0.24, -0.24, -0.88, -0.08, -0.4, -0.72, -0.4, 0.08, -1.52, 0.08, -0.4, -0.56, -0.24, -0.72, -1.2, -0.08, 0.24, -0.08, -0.4, 0.24, 0.08, -0.72, -0.72, 0.08, -0.4, -0.72, 0.24, -0.56, 0.08, 0.24, -0.08, -1.04, -0.08, -0.24, -0.08, -0.56, 0.08, -1.68, 0.4, -0.88, -0.4, 0.4, 0.24, 0.24, -0.24, -0.56, -0.24, -0.72, -0.88, -1.68, -0.24, -1.04, 0.24, -0.24, 0.56, -1.04, 0.4, -0.56, -0.08, 0.4, -0.24, -0.24, -0.24, -1.2, -0.4, -0.08, -0.24, -0.56, -0.4, -0.88, -0.4, 0.24, -0.72, -0.72, -0.88, -0.24, -0.88, -0.4, -1.52, -0.72, -0.88, -0.72, -0.72, -0.4, 0.08, -0.4, -0.72, -0.72, -0.56, 0.08, -0.4, 0.72, 0.56, 0.08, -0.08, -0.72, -0.4, -1.04, 0.56, -0.24, -0.72, -0.24, 0.08, -0.88, -1.2, 0.72, 0.24, -0.72, 0.24, -0.4, -0.08, -1.2, -0.4, -0.72, -0.4, -1.04, -0.56, -0.56, 0.08, -0.56, -0.72, -0.72, -0.72, -0.56, 0.08, 0.08, 0.24, -0.4, -0.08, -0.72, -1.04, -0.88, -0.08, -1.04, -1.2, -0.08, 0.08, -0.24, -1.2, 0.08, -0.08, -1.04, -0.56, -0.56, -0.24, -1.68, -0.56, -1.2, -1.04, -0.72, -1.2, 0.56, -0.4, -0.24, -0.24, -0.4]
top_var_reward = [3.789670170344644, 3.84, 3.5090739519138094, 3.666060555964672, 3.789670170344644, 3.5090739519138094, 3.7318092127009925, 3.84, 3.883091551843711, 3.731809212700992, 3.789670170344644, 3.883091551843711, 3.731809212700992, 3.313970428353277, 3.731809212700992, 3.313970428353277, 3.5919910913029836, 3.2, 3.789670170344644, 3.41666504065002, 3.789670170344644, 3.6660605559646715, 3.731809212700992, 3.509073951913809, 3.5090739519138094, 3.948468057360981, 3.84, 3.7896701703446434, 3.883091551843711, 3.666060555964672, 3.7318092127009925, 3.0734996339677676, 3.948468057360981, 3.5090739519138094, 3.3139704283532767, 3.7318092127009925, 3.883091551843711, 3.731809212700992, 3.5090739519138094, 3.666060555964672, 3.313970428353277, 3.50907395191381, 3.666060555964672, 3.6660605559646715, 3.5919910913029836, 3.5919910913029836, 3.84, 3.5090739519138094, 3.666060555964672, 3.789670170344644, 3.666060555964672, 3.41666504065002, 3.971095566717074, 3.41666504065002, 3.666060555964672, 3.7318092127009925, 3.5919910913029836, 3.789670170344644, 3.919183588453085, 3.5090739519138094, 3.313970428353277, 3.5090739519138094, 3.6660605559646715, 3.3139704283532767, 3.41666504065002, 3.789670170344644, 3.789670170344644, 3.41666504065002, 3.666060555964672, 3.789670170344644, 3.313970428353277, 3.731809212700992, 3.41666504065002, 3.313970428353277, 3.5090739519138094, 3.883091551843711, 3.5090739519138094, 3.5919910913029836, 3.5090739519138094, 3.7318092127009925, 3.41666504065002, 3.9871794542006764, 3.2, 3.84, 3.666060555964672, 3.2, 3.3139704283532767, 3.313970428353277, 3.591991091302984, 3.731809212700992, 3.5919910913029836, 3.789670170344644, 3.84, 3.9871794542006764, 3.591991091302984, 3.883091551843711, 3.313970428353277, 3.5919910913029836, 3.0734996339677676, 3.883091551843711, 3.2, 3.7318092127009925, 3.5090739519138094, 3.2, 3.5919910913029836, 3.5919910913029836, 3.5919910913029836, 3.919183588453085, 3.6660605559646715, 3.5090739519138094, 3.591991091302984, 3.731809212700992, 3.666060555964672, 3.84, 3.666060555964672, 3.313970428353277, 3.789670170344644, 3.789670170344644, 3.84, 3.591991091302984, 3.84, 3.666060555964672, 3.971095566717074, 3.789670170344644, 3.84, 3.789670170344644, 3.789670170344644, 3.666060555964672, 3.41666504065002, 3.6660605559646715, 3.789670170344644, 3.789670170344644, 3.7318092127009925, 3.41666504065002, 3.666060555964672, 2.932848444771737, 3.0734996339677676, 3.41666504065002, 3.5090739519138094, 3.789670170344644, 3.6660605559646715, 3.883091551843711, 3.0734996339677676, 3.591991091302984, 3.789670170344644, 3.591991091302984, 3.41666504065002, 3.84, 3.919183588453085, 2.932848444771737, 3.3139704283532763, 3.789670170344644, 3.313970428353277, 3.666060555964672, 3.5090739519138094, 3.9191835884530852, 3.6660605559646715, 3.789670170344644, 3.666060555964672, 3.883091551843711, 3.7318092127009916, 3.7318092127009925, 3.41666504065002, 3.731809212700992, 3.789670170344644, 3.789670170344644, 3.789670170344644, 3.7318092127009925, 3.41666504065002, 3.41666504065002, 3.313970428353277, 3.6660605559646715, 3.5090739519138094, 3.789670170344644, 3.883091551843711, 3.84, 3.5090739519138094, 3.883091551843711, 3.919183588453085, 3.5090739519138094, 3.41666504065002, 3.591991091302984, 3.919183588453085, 3.41666504065002, 3.5090739519138094, 3.883091551843711, 3.7318092127009925, 3.731809212700992, 3.591991091302984, 3.9871794542006764, 3.731809212700992, 3.919183588453085, 3.883091551843711, 3.789670170344644, 3.9191835884530852, 3.0734996339677676, 3.6660605559646715, 3.591991091302984, 3.5919910913029836, 3.666060555964672]
top_mavg_steps = [3.32, 3.28, 3.48, 3.4, 3.32, 3.48, 3.36, 3.28, 3.24, 3.36, 3.32, 3.24, 3.36, 3.56, 3.36, 3.56, 3.44, 3.6, 3.32, 3.52, 3.32, 3.4, 3.36, 3.48, 3.48, 3.16, 3.28, 3.32, 3.24, 3.4, 3.36, 3.64, 3.16, 3.48, 3.56, 3.36, 3.24, 3.36, 3.48, 3.4, 3.56, 3.48, 3.4, 3.4, 3.44, 3.44, 3.28, 3.48, 3.4, 3.32, 3.4, 3.52, 3.12, 3.52, 3.4, 3.36, 3.44, 3.32, 3.2, 3.48, 3.56, 3.48, 3.4, 3.56, 3.52, 3.32, 3.32, 3.52, 3.4, 3.32, 3.56, 3.36, 3.52, 3.56, 3.48, 3.24, 3.48, 3.44, 3.48, 3.36, 3.52, 3.08, 3.6, 3.28, 3.4, 3.6, 3.56, 3.56, 3.44, 3.36, 3.44, 3.32, 3.28, 3.08, 3.44, 3.24, 3.56, 3.44, 3.64, 3.24, 3.6, 3.36, 3.48, 3.6, 3.44, 3.44, 3.44, 3.2, 3.4, 3.48, 3.44, 3.36, 3.4, 3.28, 3.4, 3.56, 3.32, 3.32, 3.28, 3.44, 3.28, 3.4, 3.12, 3.32, 3.28, 3.32, 3.32, 3.4, 3.52, 3.4, 3.32, 3.32, 3.36, 3.52, 3.4, 3.68, 3.64, 3.52, 3.48, 3.32, 3.4, 3.24, 3.64, 3.44, 3.32, 3.44, 3.52, 3.28, 3.2, 3.68, 3.56, 3.32, 3.56, 3.4, 3.48, 3.2, 3.4, 3.32, 3.4, 3.24, 3.36, 3.36, 3.52, 3.36, 3.32, 3.32, 3.32, 3.36, 3.52, 3.52, 3.56, 3.4, 3.48, 3.32, 3.24, 3.28, 3.48, 3.24, 3.2, 3.48, 3.52, 3.44, 3.2, 3.52, 3.48, 3.24, 3.36, 3.36, 3.44, 3.08, 3.36, 3.2, 3.24, 3.32, 3.2, 3.64, 3.4, 3.44, 3.44, 3.4]
top_var_steps = [0.947417542586161, 0.96, 0.8772684879784524, 0.916515138991168, 0.947417542586161, 0.8772684879784524, 0.9329523031752481, 0.96, 0.9707728879609276, 0.932952303175248, 0.947417542586161, 0.9707728879609276, 0.932952303175248, 0.8284926070883191, 0.932952303175248, 0.8284926070883191, 0.8979977728257459, 0.8, 0.947417542586161, 0.854166260162505, 0.947417542586161, 0.916515138991168, 0.932952303175248, 0.8772684879784522, 0.8772684879784524, 0.9871170143402452, 0.96, 0.947417542586161, 0.9707728879609276, 0.9165151389911681, 0.9329523031752481, 0.7683749084919419, 0.9871170143402452, 0.8772684879784524, 0.8284926070883192, 0.9329523031752481, 0.9707728879609276, 0.932952303175248, 0.8772684879784524, 0.916515138991168, 0.8284926070883191, 0.8772684879784525, 0.916515138991168, 0.916515138991168, 0.8979977728257459, 0.8979977728257459, 0.96, 0.8772684879784524, 0.916515138991168, 0.9474175425861608, 0.9165151389911681, 0.854166260162505, 0.9927738916792684, 0.854166260162505, 0.916515138991168, 0.9329523031752481, 0.8979977728257459, 0.947417542586161, 0.9797958971132712, 0.8772684879784524, 0.8284926070883191, 0.8772684879784524, 0.9165151389911681, 0.8284926070883192, 0.854166260162505, 0.947417542586161, 0.947417542586161, 0.854166260162505, 0.916515138991168, 0.947417542586161, 0.8284926070883191, 0.932952303175248, 0.854166260162505, 0.8284926070883191, 0.8772684879784524, 0.9707728879609276, 0.8772684879784524, 0.8979977728257459, 0.8772684879784524, 0.9329523031752481, 0.854166260162505, 0.9967948635501691, 0.8, 0.96, 0.916515138991168, 0.8, 0.8284926070883192, 0.8284926070883191, 0.897997772825746, 0.9329523031752481, 0.8979977728257459, 0.947417542586161, 0.96, 0.9967948635501691, 0.897997772825746, 0.9707728879609278, 0.8284926070883192, 0.8979977728257459, 0.7683749084919419, 0.9707728879609276, 0.8, 0.9329523031752481, 0.8772684879784524, 0.8, 0.8979977728257459, 0.8979977728257459, 0.8979977728257459, 0.9797958971132712, 0.9165151389911681, 0.8772684879784524, 0.897997772825746, 0.932952303175248, 0.916515138991168, 0.96, 0.916515138991168, 0.8284926070883191, 0.9474175425861608, 0.9474175425861608, 0.96, 0.897997772825746, 0.96, 0.9165151389911681, 0.9927738916792685, 0.947417542586161, 0.96, 0.947417542586161, 0.947417542586161, 0.916515138991168, 0.854166260162505, 0.916515138991168, 0.9474175425861608, 0.947417542586161, 0.9329523031752481, 0.854166260162505, 0.916515138991168, 0.7332121111929343, 0.7683749084919419, 0.854166260162505, 0.8772684879784524, 0.947417542586161, 0.9165151389911681, 0.9707728879609276, 0.7683749084919419, 0.897997772825746, 0.947417542586161, 0.897997772825746, 0.854166260162505, 0.96, 0.9797958971132712, 0.7332121111929344, 0.8284926070883191, 0.947417542586161, 0.8284926070883191, 0.916515138991168, 0.8772684879784524, 0.9797958971132712, 0.9165151389911681, 0.947417542586161, 0.916515138991168, 0.9707728879609276, 0.932952303175248, 0.9329523031752481, 0.854166260162505, 0.932952303175248, 0.947417542586161, 0.947417542586161, 0.9474175425861608, 0.9329523031752481, 0.854166260162505, 0.854166260162505, 0.8284926070883191, 0.9165151389911681, 0.8772684879784524, 0.947417542586161, 0.9707728879609276, 0.96, 0.8772684879784524, 0.9707728879609276, 0.9797958971132712, 0.8772684879784524, 0.854166260162505, 0.897997772825746, 0.9797958971132712, 0.854166260162505, 0.8772684879784524, 0.9707728879609276, 0.9329523031752481, 0.932952303175248, 0.897997772825746, 0.9967948635501691, 0.932952303175248, 0.9797958971132712, 0.9707728879609276, 0.947417542586161, 0.9797958971132712, 0.7683749084919419, 0.9165151389911681, 0.897997772825746, 0.8979977728257459, 0.916515138991168]
top_end_count = [0.34, 0.36, 0.26, 0.3, 0.34, 0.26, 0.32, 0.36, 0.38, 0.32, 0.34, 0.38, 0.32, 0.22, 0.32, 0.22, 0.28, 0.2, 0.34, 0.24, 0.34, 0.3, 0.32, 0.26, 0.26, 0.42, 0.36, 0.34, 0.38, 0.3, 0.32, 0.18, 0.42, 0.26, 0.22, 0.32, 0.38, 0.32, 0.26, 0.3, 0.22, 0.26, 0.3, 0.3, 0.28, 0.28, 0.36, 0.26, 0.3, 0.34, 0.3, 0.24, 0.44, 0.24, 0.3, 0.32, 0.28, 0.34, 0.4, 0.26, 0.22, 0.26, 0.3, 0.22, 0.24, 0.34, 0.34, 0.24, 0.3, 0.34, 0.22, 0.32, 0.24, 0.22, 0.26, 0.38, 0.26, 0.28, 0.26, 0.32, 0.24, 0.46, 0.2, 0.36, 0.3, 0.2, 0.22, 0.22, 0.28, 0.32, 0.28, 0.34, 0.36, 0.46, 0.28, 0.38, 0.22, 0.28, 0.18, 0.38, 0.2, 0.32, 0.26, 0.2, 0.28, 0.28, 0.28, 0.4, 0.3, 0.26, 0.28, 0.32, 0.3, 0.36, 0.3, 0.22, 0.34, 0.34, 0.36, 0.28, 0.36, 0.3, 0.44, 0.34, 0.36, 0.34, 0.34, 0.3, 0.24, 0.3, 0.34, 0.34, 0.32, 0.24, 0.3, 0.16, 0.18, 0.24, 0.26, 0.34, 0.3, 0.38, 0.18, 0.28, 0.34, 0.28, 0.24, 0.36, 0.4, 0.16, 0.22, 0.34, 0.22, 0.3, 0.26, 0.4, 0.3, 0.34, 0.3, 0.38, 0.32, 0.32, 0.24, 0.32, 0.34, 0.34, 0.34, 0.32, 0.24, 0.24, 0.22, 0.3, 0.26, 0.34, 0.38, 0.36, 0.26, 0.38, 0.4, 0.26, 0.24, 0.28, 0.4, 0.24, 0.26, 0.38, 0.32, 0.32, 0.28, 0.46, 0.32, 0.4, 0.38, 0.34, 0.4, 0.18, 0.3, 0.28, 0.28, 0.3]

middle_mavg_reward = [-0.52, -0.44, -0.36, -0.52, -0.4, -0.04, -1.04, -1.16, -0.46, -0.1, -0.32, -0.22, -0.68, -0.44, -0.66, -0.18, -0.74, -0.4, -0.28, -0.08, -0.2, -0.34, -0.36, -0.82, -0.6, -0.62, -0.54, -0.64, -0.1, -0.26, -0.32, -0.82, -0.1, -0.4, -0.88, -0.28, -0.5, -0.18, -0.5, -0.84, -0.92, -0.18, -0.48, -0.3, -1.2, -0.72, -0.58, -0.64, 0.0, -0.98, -0.78, -0.6, -0.78, -0.48, -0.16, -0.2, -0.08, -0.82, -1.08, -0.38, -1.12, -0.7, -0.52, 0.1, -0.74, -0.54, -0.46, -0.92, -0.12, -0.84, -0.94, -0.5, -0.22, 0.08, 0.06, -1.02, -0.7, -0.36, -0.38, -0.2, -0.5, -0.54, -0.54, -0.36, -0.38, -0.14, -0.76, -0.58, -0.52, -0.62, -0.4, -0.76, -0.38, -0.48, -0.64, -0.6, -1.14, -0.44, -0.66, -1.28, -0.18, -0.72, -0.32, -1.12, -0.16, -0.16, -1.0, 0.18, -0.3, -0.18, -0.88, -0.14, -0.78, -0.12, -0.04, -0.74, -0.76, -0.12, -0.44, -0.06, -0.84, -0.38, -1.38, -1.0, -0.3, -0.38, -0.72, -0.7, 0.12, -0.2, -0.24, -0.18, -0.42, -0.56, -0.36, -0.38, -0.42, -0.44, -0.66, 0.04, -1.06, -0.66, -0.12, -0.58, -0.54, -0.5, -1.3, -0.54, -0.98, -0.4, -0.64, -0.3, -0.74, -0.58, -0.64, 0.04, -0.46, 0.08, -0.48, -0.26, -0.56, -0.26, -0.88, -0.34, -0.8, -0.2, 0.0, -0.48, -0.86, -0.44, 0.26, -0.26, -0.62, -0.26, -0.4, -0.66, -0.34, -0.44, -0.1, -0.44, -0.44, -0.38, -0.88, -0.32, -0.74, -0.7, -0.86, -0.74, 0.16, -0.48, -1.1, -0.86, -0.42, -0.56, -1.56, -0.34, -0.44, -0.8, -0.24, -0.02]
middle_var_reward = [2.193080025899648, 1.961224107541002, 1.9975985582694038, 2.1930800258996483, 1.9798989873223327, 1.482700239428051, 2.821063629200873, 2.9486267990371378, 2.22, 1.4594519519326425, 2.0143485299222674, 1.7582946283259815, 2.361694307060082, 2.228542124349459, 2.3715817506466017, 1.7741476826916074, 2.5597656142701815, 1.9798989873223327, 1.732512626216617, 1.4675149062275312, 1.766352173265569, 2.0060907257649143, 1.9975985582694038, 2.5194443831924533, 2.4000000000000004, 2.3907321054438535, 2.183666641225258, 2.381260170581955, 1.4594519519326423, 1.7413787640832195, 2.0143485299222674, 2.519444383192453, 1.4594519519326425, 1.9798989873223327, 2.7028873450441844, 1.732512626216617, 2.202271554554524, 1.7741476826916074, 2.202271554554524, 2.5088642848906755, 2.6820887382784337, 1.7741476826916076, 2.2112439937736403, 2.022374841615669, 2.9257477676655586, 2.569357896440276, 2.4090662091358137, 2.605839595984373, 1.4966629547095767, 2.6494527736874267, 2.54, 2.4000000000000004, 2.54, 2.211243993773641, 1.781684596105607, 1.766352173265569, 1.8093092604637828, 2.519444383192453, 2.7988569095257443, 1.9888690253508399, 2.775896251663595, 2.3515952032609695, 2.193080025899648, 1.0999999999999999, 2.330750951946604, 2.183666641225258, 2.22, 2.6820887382784337, 1.4510685717773644, 2.722939588018801, 2.6714041251746243, 2.202271554554524, 1.758294628325981, 1.092520022699813, 1.0846197490365, 2.6267089675104853, 2.5787593916455256, 1.9975985582694038, 1.9888690253508399, 1.766352173265569, 2.202271554554524, 2.183666641225258, 2.183666641225258, 1.9975985582694038, 1.9888690253508399, 1.4423591785682233, 2.5499803920814763, 2.409066209135814, 2.193080025899648, 2.3907321054438535, 1.9798989873223327, 2.5499803920814768, 1.9888690253508399, 2.2112439937736403, 2.381260170581955, 2.4000000000000004, 2.7641273487305176, 2.228542124349459, 2.3715817506466017, 2.8777769197767915, 1.7741476826916074, 2.3412817002659034, 2.0143485299222674, 2.7758962516635957, 1.7816845961056067, 1.781684596105607, 2.842534080710379, 1.1258774356030057, 1.7233687939614089, 1.7741476826916076, 2.7028873450441844, 1.442359178568223, 2.54, 1.4510685717773641, 1.482700239428051, 2.5597656142701815, 2.5499803920814763, 1.7959955456514918, 2.228542124349459, 1.4752626884728024, 2.5088642848906755, 1.9888690253508399, 3.0059274775017446, 2.6381811916545836, 1.7233687939614089, 1.9888690253508399, 2.3412817002659034, 2.3515952032609695, 1.1070682002478438, 1.766352173265569, 1.7499714283381886, 1.7741476826916074, 1.970685160039523, 2.417933001553186, 1.9975985582694038, 1.9888690253508399, 1.970685160039523, 2.228542124349459, 2.3715817506466017, 1.5094369811290567, 2.810053380275898, 2.3715817506466017, 1.4510685717773644, 2.4090662091358137, 2.183666641225258, 2.202271554554524, 2.8653097563788807, 2.1836666412252583, 2.649452773687427, 1.9798989873223327, 2.3812601705819545, 2.022374841615669, 2.5597656142701815, 2.4090662091358137, 2.3812601705819545, 1.076289923765897, 1.951512234140488, 1.092520022699813, 1.9415457759218557, 2.0377438504385186, 2.1740285186721904, 1.7413787640832192, 2.7028873450441844, 2.0060907257649143, 2.5298221281347035, 1.7663521732655691, 1.4966629547095767, 2.2112439937736403, 2.4980792621532246, 2.228542124349459, 0.4386342439892262, 1.7413787640832192, 2.390732105443853, 2.0377438504385186, 2.2449944320643644, 2.371581750646602, 1.704230031421815, 2.228542124349459, 1.4594519519326425, 1.9612241075410017, 2.228542124349459, 1.9888690253508399, 2.48708664907357, 2.0143485299222674, 2.5597656142701815, 2.5787593916455256, 2.4980792621532246, 2.330750951946604, 1.1199999999999999, 2.2112439937736403, 2.787471972953271, 2.7130057132265684, 2.236872817126624, 2.174028518672191, 3.073499633967767, 2.0060907257649143, 2.228542124349459, 2.5298221281347035, 1.7499714283381886, 1.4898322053170956]
middle_mavg_steps = [5.52, 5.64, 5.56, 5.52, 5.6, 5.64, 5.24, 5.16, 5.46, 5.7, 5.52, 5.62, 5.48, 5.44, 5.46, 5.58, 5.34, 5.6, 5.68, 5.68, 5.6, 5.54, 5.56, 5.42, 5.4, 5.42, 5.54, 5.44, 5.7, 5.66, 5.52, 5.42, 5.7, 5.6, 5.28, 5.68, 5.5, 5.58, 5.5, 5.44, 5.32, 5.58, 5.48, 5.5, 5.2, 5.32, 5.38, 5.24, 5.6, 5.38, 5.38, 5.4, 5.38, 5.48, 5.56, 5.6, 5.48, 5.42, 5.28, 5.58, 5.32, 5.5, 5.52, 5.7, 5.54, 5.54, 5.46, 5.32, 5.72, 5.24, 5.34, 5.5, 5.62, 5.72, 5.74, 5.42, 5.3, 5.56, 5.58, 5.6, 5.5, 5.54, 5.54, 5.56, 5.58, 5.74, 5.36, 5.38, 5.52, 5.42, 5.6, 5.36, 5.58, 5.48, 5.44, 5.4, 5.34, 5.44, 5.46, 5.28, 5.58, 5.52, 5.52, 5.32, 5.56, 5.56, 5.2, 5.62, 5.7, 5.58, 5.28, 5.74, 5.38, 5.72, 5.64, 5.34, 5.36, 5.52, 5.44, 5.66, 5.44, 5.58, 5.18, 5.4, 5.7, 5.58, 5.52, 5.5, 5.68, 5.6, 5.64, 5.58, 5.62, 5.36, 5.56, 5.58, 5.62, 5.44, 5.46, 5.56, 5.26, 5.46, 5.72, 5.38, 5.54, 5.5, 5.3, 5.54, 5.38, 5.6, 5.44, 5.5, 5.34, 5.38, 5.44, 5.76, 5.66, 5.72, 5.68, 5.46, 5.56, 5.66, 5.28, 5.54, 5.4, 5.6, 5.6, 5.48, 5.46, 5.44, 5.74, 5.66, 5.42, 5.46, 5.4, 5.46, 5.74, 5.44, 5.7, 5.64, 5.44, 5.58, 5.48, 5.52, 5.34, 5.3, 5.46, 5.54, 5.64, 5.48, 5.3, 5.26, 5.42, 5.56, 5.16, 5.54, 5.44, 5.4, 5.64, 5.62]
middle_var_steps = [0.9217374897442331, 0.8428523002282192, 0.8522910301065006, 0.9217374897442331, 0.848528137423857, 0.6858571279792898, 1.123565752415051, 1.1551623262554922, 0.9210863151735564, 0.6708203932499369, 0.854166260162505, 0.7717512552629895, 0.9846826900072937, 0.9199999999999999, 0.9840731680114035, 0.7769169839822013, 1.0316976301223144, 0.848528137423857, 0.76, 0.6764613810115105, 0.7745966692414834, 0.8534635317340747, 0.8522910301065005, 1.0409610943738483, 0.9797958971132712, 0.9816312953446421, 0.9210863151735563, 0.9830564581955606, 0.6708203932499369, 0.7644605941446557, 0.854166260162505, 1.0409610943738483, 0.6708203932499369, 0.848528137423857, 1.0777754868245986, 0.7599999999999999, 0.9219544457292888, 0.7769169839822013, 0.9219544457292888, 1.0423051376636305, 1.085172797300043, 0.7769169839822013, 0.9217374897442331, 0.8544003745317531, 1.1661903789690602, 1.0283968105745953, 0.9775479527879949, 1.011137972781163, 0.6928203230275509, 1.0934349546269315, 1.037111372997134, 0.9797958971132712, 1.037111372997134, 0.9217374897442331, 0.7787168933572713, 0.7745966692414833, 0.7807688518377254, 1.0409610943738483, 1.1320777358467924, 0.8506468127254696, 1.1391224692718513, 0.9848857801796105, 0.9217374897442331, 0.5744562646538028, 0.9840731680114033, 0.9210863151735563, 0.9210863151735564, 1.085172797300043, 0.664529909033446, 1.0688311372709909, 1.0883014288330233, 0.9219544457292888, 0.7717512552629896, 0.567097875150313, 0.5589275444992848, 1.0970870521521983, 1.02469507659596, 0.8522910301065005, 0.8506468127254696, 0.7745966692414834, 0.9219544457292888, 0.9210863151735563, 0.9210863151735563, 0.8522910301065006, 0.8506468127254695, 0.6575712889109437, 1.034601372510205, 0.977547952787995, 0.9217374897442331, 0.9816312953446421, 0.8485281374238571, 1.034601372510205, 0.8506468127254695, 0.921737489744233, 0.9830564581955606, 0.9797958971132712, 1.1421033228215387, 0.92, 0.9840731680114035, 1.1838918869558994, 0.7769169839822013, 0.9846826900072938, 0.854166260162505, 1.1391224692718513, 0.7787168933572713, 0.7787168933572713, 1.1135528725660044, 0.5963220606350229, 0.7549834435270749, 0.7769169839822013, 1.0777754868245986, 0.6575712889109439, 1.037111372997134, 0.664529909033446, 0.6858571279792899, 1.0316976301223144, 1.034601372510205, 0.7807688518377254, 0.92, 0.681469001496033, 1.0423051376636305, 0.8506468127254696, 1.2114454176726244, 1.0954451150103321, 0.7549834435270749, 0.8506468127254695, 0.9846826900072937, 0.9848857801796105, 0.5810335618533581, 0.7745966692414834, 0.7683749084919418, 0.7769169839822013, 0.845931439302264, 0.9748846085563152, 0.8522910301065008, 0.8506468127254696, 0.845931439302264, 0.92, 0.9840731680114035, 0.6974238309665078, 1.128007092176286, 0.9840731680114035, 0.664529909033446, 0.977547952787995, 0.9210863151735563, 0.9219544457292888, 1.1874342087037917, 0.9210863151735564, 1.0934349546269315, 0.848528137423857, 0.9830564581955606, 0.8544003745317531, 1.0316976301223144, 0.9775479527879949, 0.9830564581955605, 0.5499090833947009, 0.8392854103342915, 0.567097875150313, 0.8352245207128441, 0.8534635317340747, 0.9199999999999999, 0.7644605941446557, 1.0777754868245983, 0.8534635317340747, 1.0392304845413265, 0.7745966692414834, 0.6928203230275509, 0.9217374897442331, 1.0432641084595982, 0.9199999999999999, 0.4386342439892262, 0.7644605941446557, 0.9816312953446421, 0.8534635317340747, 0.916515138991168, 0.9840731680114035, 0.7432361670424817, 0.9199999999999999, 0.6708203932499369, 0.8428523002282192, 0.92, 0.8506468127254695, 1.0438390680559912, 0.854166260162505, 1.0316976301223144, 1.02469507659596, 1.0432641084595982, 0.9840731680114035, 0.5919459434779497, 0.921737489744233, 1.1357816691600546, 1.0734989520255713, 0.9184770002564027, 0.92, 1.2547509713086498, 0.8534635317340745, 0.92, 1.0392304845413263, 0.7683749084919418, 0.6896375859826667]
middle_end_count = [0.1, 0.08, 0.08, 0.1, 0.08, 0.04, 0.18, 0.2, 0.1, 0.04, 0.08, 0.06, 0.12, 0.1, 0.12, 0.06, 0.14, 0.08, 0.06, 0.04, 0.06, 0.08, 0.08, 0.14, 0.12, 0.12, 0.1, 0.12, 0.04, 0.06, 0.08, 0.14, 0.04, 0.08, 0.16, 0.06, 0.1, 0.06, 0.1, 0.14, 0.16, 0.06, 0.1, 0.08, 0.2, 0.14, 0.12, 0.14, 0.04, 0.16, 0.14, 0.12, 0.14, 0.1, 0.06, 0.06, 0.06, 0.14, 0.18, 0.08, 0.18, 0.12, 0.1, 0.02, 0.12, 0.1, 0.1, 0.16, 0.04, 0.16, 0.16, 0.1, 0.06, 0.02, 0.02, 0.16, 0.14, 0.08, 0.08, 0.06, 0.1, 0.1, 0.1, 0.08, 0.08, 0.04, 0.14, 0.12, 0.1, 0.12, 0.08, 0.14, 0.08, 0.1, 0.12, 0.12, 0.18, 0.1, 0.12, 0.2, 0.06, 0.12, 0.08, 0.18, 0.06, 0.06, 0.18, 0.02, 0.06, 0.06, 0.16, 0.04, 0.14, 0.04, 0.04, 0.14, 0.14, 0.06, 0.1, 0.04, 0.14, 0.08, 0.22, 0.16, 0.06, 0.08, 0.12, 0.12, 0.02, 0.06, 0.06, 0.06, 0.08, 0.12, 0.08, 0.08, 0.08, 0.1, 0.12, 0.04, 0.18, 0.12, 0.04, 0.12, 0.1, 0.1, 0.2, 0.1, 0.16, 0.08, 0.12, 0.08, 0.14, 0.12, 0.12, 0.02, 0.08, 0.02, 0.08, 0.08, 0.1, 0.06, 0.16, 0.08, 0.14, 0.06, 0.04, 0.1, 0.14, 0.1, 0.0, 0.06, 0.12, 0.08, 0.1, 0.12, 0.06, 0.1, 0.04, 0.08, 0.1, 0.08, 0.14, 0.08, 0.14, 0.14, 0.14, 0.12, 0.02, 0.1, 0.18, 0.16, 0.1, 0.1, 0.24, 0.08, 0.1, 0.14, 0.06, 0.04]

bottom_mavg_reward = [-1.62, -1.74, -1.46, -1.7, -1.64, -1.62, -1.56, -1.76, -1.78, -1.48, -1.58, -1.36, -1.74, -1.46, -1.62, -1.62, -1.62, -1.64, -1.48, -1.68, -1.72, -1.68, -1.7, -1.6, -1.66, -1.58, -1.54, -1.58, -1.6, -1.4, -1.58, -1.72, -1.68, -1.58, -1.48, -1.68, -1.66, -1.7, -1.62, -1.52, -1.56, -1.62, -1.54, -1.68, -1.52, -1.76, -1.48, -1.68, -1.62, -1.44, -1.56, -1.5, -1.56, -1.72, -1.62, -1.48, -1.62, -1.62, -1.54, -1.64, -1.74, -1.44, -1.74, -1.64, -1.34, -1.44, -1.6, -1.76, -1.46, -1.7, -1.48, -1.64, -1.58, -1.4, -1.58, -1.74, -1.6, -1.72, -1.62, -1.6, -1.66, -1.76, -1.84, -1.8, -1.72, -1.5, -1.68, -1.32, -1.76, -1.66, -1.6, -1.68, -1.66, -1.56, -1.58, -1.68, -1.62, -1.68, -1.74, -1.82, -1.7, -1.62, -1.5, -1.6, -1.62, -1.56, -1.72, -1.54, -1.6, -1.64, -1.7, -1.6, -1.44, -1.66, -1.74, -1.76, -1.74, -1.38, -1.66, -1.68, -1.52, -1.6, -1.48, -1.66, -1.76, -1.62, -1.5, -1.62, -1.76, -1.46, -1.58, -1.6, -1.74, -1.54, -1.5, -1.7, -1.66, -1.58, -1.6, -1.52, -1.56, -1.68, -1.68, -1.64, -1.74, -1.58, -1.58, -1.54, -1.56, -1.5, -1.66, -1.7, -1.64, -1.66, -1.58, -1.68, -1.62, -1.52, -1.52, -1.58, -1.54, -1.7, -1.6, -1.6, -1.62, -1.58, -1.72, -1.82, -1.78, -1.6, -1.56, -1.78, -1.72, -1.62, -1.64, -1.76, -1.68, -1.5, -1.66, -1.48, -1.56, -1.64, -1.5, -1.72, -1.58, -1.44, -1.7, -1.64, -1.62, -1.62, -1.66, -1.4, -1.5, -1.6, -1.6, -1.7, -1.6, -1.66, -1.62, -1.64]
bottom_var_reward = [0.5617828762075255, 0.5589275444992848, 0.6696267617113282, 0.6082762530298219, 0.5919459434779497, 0.66, 0.7525955088890712, 0.5122499389946279, 0.54, 0.6705221845696084, 0.6660330322138686, 0.656048778674269, 0.4386342439892262, 0.7539230729988305, 0.745385806143369, 0.6289674077406555, 0.6289674077406555, 0.5919459434779497, 0.7547184905645282, 0.5810335618533581, 0.6013318551349163, 0.6462197768561405, 0.6082762530298219, 0.66332495807108, 0.6200000000000001, 0.7236021006050218, 0.7539230729988305, 0.7236021006050218, 0.6324555320336759, 0.6928203230275509, 0.6954135460285484, 0.530659966456864, 0.6144916598294886, 0.6352952069707437, 0.8059776671843955, 0.6144916598294886, 0.6200000000000001, 0.5744562646538028, 0.6289674077406556, 0.7547184905645282, 0.7255342858886822, 0.7180529228406497, 0.7269112738154498, 0.5455272678794343, 0.7547184905645283, 0.5122499389946279, 0.7547184905645283, 0.581033561853358, 0.66, 0.7255342858886822, 0.7255342858886822, 0.7280109889280518, 0.6374950980203692, 0.567097875150313, 0.5617828762075255, 0.7547184905645282, 0.66, 0.66, 0.6696267617113283, 0.5919459434779497, 0.5219195340279956, 0.6974238309665078, 0.5936328831862332, 0.6858571279792898, 0.7901898506055365, 0.6974238309665078, 0.7211102550927979, 0.5122499389946279, 0.7539230729988305, 0.5385164807134504, 0.7547184905645282, 0.6560487786742689, 0.6954135460285485, 0.7211102550927979, 0.6352952069707437, 0.5589275444992848, 0.66332495807108, 0.567097875150313, 0.6289674077406556, 0.6324555320336759, 0.6514598989960932, 0.5499090833947008, 0.463033476111609, 0.4898979485566356, 0.530659966456864, 0.7280109889280518, 0.6144916598294886, 0.8109253973085317, 0.5122499389946279, 0.5517245689653488, 0.6633249580710799, 0.6144916598294886, 0.5868560300448484, 0.7255342858886822, 0.6029925372672535, 0.581033561853358, 0.596322060635023, 0.6144916598294886, 0.5219195340279956, 0.43312815655415426, 0.5744562646538028, 0.6289674077406556, 0.7280109889280518, 0.6324555320336759, 0.6289674077406555, 0.6974238309665077, 0.6013318551349163, 0.72691127381545, 0.6633249580710799, 0.6858571279792898, 0.6082762530298219, 0.6324555320336759, 0.7787168933572715, 0.6514598989960934, 0.5936328831862332, 0.47159304490206383, 0.5219195340279955, 0.7453858061433689, 0.5868560300448484, 0.6462197768561404, 0.699714227381436, 0.66332495807108, 0.72773621594641, 0.5517245689653488, 0.47159304490206383, 0.596322060635023, 0.6403124237432849, 0.7180529228406498, 0.5499090833947008, 0.7799999999999999, 0.6660330322138686, 0.6633249580710799, 0.5589275444992848, 0.6988562083862458, 0.7549834435270749, 0.5744562646538027, 0.5517245689653488, 0.7236021006050218, 0.7211102550927979, 0.699714227381436, 0.7255342858886822, 0.581033561853358, 0.6144916598294886, 0.5571355310873648, 0.5219195340279956, 0.6954135460285485, 0.6029925372672534, 0.6390618123468182, 0.6681317235396026, 0.7280109889280518, 0.6200000000000001, 0.5385164807134504, 0.6858571279792898, 0.62, 0.6660330322138686, 0.6462197768561404, 0.6289674077406556, 0.72773621594641, 0.64, 0.6954135460285485, 0.6696267617113282, 0.458257569495584, 0.6324555320336759, 0.66332495807108, 0.6896375859826668, 0.7236021006050218, 0.567097875150313, 0.4770744176750625, 0.5399999999999999, 0.7211102550927979, 0.7255342858886822, 0.54, 0.567097875150313, 0.7180529228406497, 0.656048778674269, 0.5122499389946279, 0.5455272678794342, 0.6708203932499369, 0.6514598989960932, 0.7807688518377254, 0.7525955088890712, 0.6248199740725323, 0.7549834435270749, 0.530659966456864, 0.7507329751649385, 0.7255342858886822, 0.6082762530298219, 0.5571355310873647, 0.66, 0.6289674077406555, 0.6514598989960932, 0.7483314773547883, 0.7, 0.6633249580710799, 0.6633249580710799, 0.5385164807134505, 0.6324555320336759, 0.6514598989960932, 0.66, 0.5919459434779497]
bottom_mavg_steps = [7.62, 7.74, 7.46, 7.7, 7.64, 7.62, 7.56, 7.76, 7.78, 7.48, 7.58, 7.36, 7.74, 7.46, 7.62, 7.62, 7.62, 7.64, 7.48, 7.68, 7.72, 7.68, 7.7, 7.6, 7.66, 7.58, 7.54, 7.58, 7.6, 7.4, 7.58, 7.72, 7.68, 7.58, 7.48, 7.68, 7.66, 7.7, 7.62, 7.52, 7.56, 7.62, 7.54, 7.68, 7.52, 7.76, 7.48, 7.68, 7.62, 7.44, 7.56, 7.5, 7.56, 7.72, 7.62, 7.48, 7.62, 7.62, 7.54, 7.64, 7.74, 7.44, 7.74, 7.64, 7.34, 7.44, 7.6, 7.76, 7.46, 7.7, 7.48, 7.64, 7.58, 7.4, 7.58, 7.74, 7.6, 7.72, 7.62, 7.6, 7.66, 7.76, 7.84, 7.8, 7.72, 7.5, 7.68, 7.32, 7.76, 7.66, 7.6, 7.68, 7.66, 7.56, 7.58, 7.68, 7.62, 7.68, 7.74, 7.82, 7.7, 7.62, 7.5, 7.6, 7.62, 7.56, 7.72, 7.54, 7.6, 7.64, 7.7, 7.6, 7.44, 7.66, 7.74, 7.76, 7.74, 7.38, 7.66, 7.68, 7.52, 7.6, 7.48, 7.66, 7.76, 7.62, 7.5, 7.62, 7.76, 7.46, 7.58, 7.6, 7.74, 7.54, 7.5, 7.7, 7.66, 7.58, 7.6, 7.52, 7.56, 7.68, 7.68, 7.64, 7.74, 7.58, 7.58, 7.54, 7.56, 7.5, 7.66, 7.7, 7.64, 7.66, 7.58, 7.68, 7.62, 7.52, 7.52, 7.58, 7.54, 7.7, 7.6, 7.6, 7.62, 7.58, 7.72, 7.82, 7.78, 7.6, 7.56, 7.78, 7.72, 7.62, 7.64, 7.76, 7.68, 7.5, 7.66, 7.48, 7.56, 7.64, 7.5, 7.72, 7.58, 7.44, 7.7, 7.64, 7.62, 7.62, 7.66, 7.4, 7.5, 7.6, 7.6, 7.7, 7.6, 7.66, 7.62, 7.64]
bottom_var_steps = [0.5617828762075255, 0.5589275444992848, 0.6696267617113282, 0.6082762530298219, 0.5919459434779497, 0.66, 0.7525955088890712, 0.512249938994628, 0.54, 0.6705221845696083, 0.6660330322138686, 0.656048778674269, 0.4386342439892262, 0.7539230729988305, 0.745385806143369, 0.6289674077406555, 0.6289674077406555, 0.5919459434779497, 0.7547184905645282, 0.5810335618533581, 0.6013318551349163, 0.6462197768561405, 0.6082762530298219, 0.6633249580710799, 0.62, 0.7236021006050218, 0.7539230729988305, 0.7236021006050218, 0.6324555320336758, 0.6928203230275509, 0.6954135460285484, 0.530659966456864, 0.6144916598294887, 0.6352952069707437, 0.8059776671843955, 0.6144916598294886, 0.62, 0.5744562646538028, 0.6289674077406556, 0.7547184905645282, 0.7255342858886822, 0.7180529228406497, 0.7269112738154498, 0.5455272678794342, 0.7547184905645282, 0.512249938994628, 0.7547184905645282, 0.581033561853358, 0.66, 0.7255342858886823, 0.7255342858886822, 0.7280109889280518, 0.6374950980203692, 0.5670978751503131, 0.5617828762075255, 0.7547184905645282, 0.66, 0.66, 0.6696267617113283, 0.5919459434779497, 0.5219195340279956, 0.6974238309665078, 0.5936328831862332, 0.6858571279792899, 0.7901898506055364, 0.6974238309665078, 0.7211102550927979, 0.512249938994628, 0.7539230729988305, 0.5385164807134504, 0.7547184905645282, 0.656048778674269, 0.6954135460285485, 0.7211102550927979, 0.6352952069707437, 0.5589275444992848, 0.6633249580710799, 0.567097875150313, 0.6289674077406556, 0.6324555320336759, 0.6514598989960932, 0.5499090833947009, 0.463033476111609, 0.4898979485566356, 0.530659966456864, 0.7280109889280518, 0.6144916598294886, 0.8109253973085319, 0.512249938994628, 0.5517245689653488, 0.6633249580710799, 0.6144916598294886, 0.5868560300448484, 0.7255342858886822, 0.6029925372672535, 0.581033561853358, 0.596322060635023, 0.6144916598294886, 0.5219195340279956, 0.43312815655415426, 0.5744562646538028, 0.6289674077406556, 0.7280109889280518, 0.6324555320336758, 0.6289674077406555, 0.6974238309665077, 0.6013318551349164, 0.72691127381545, 0.6633249580710799, 0.6858571279792899, 0.6082762530298219, 0.6324555320336759, 0.7787168933572715, 0.6514598989960932, 0.5936328831862331, 0.47159304490206383, 0.5219195340279956, 0.7453858061433689, 0.5868560300448484, 0.6462197768561405, 0.699714227381436, 0.6633249580710799, 0.72773621594641, 0.5517245689653488, 0.47159304490206383, 0.596322060635023, 0.6403124237432849, 0.7180529228406498, 0.5499090833947009, 0.7799999999999999, 0.6660330322138686, 0.6633249580710799, 0.5589275444992847, 0.6988562083862458, 0.7549834435270749, 0.5744562646538028, 0.5517245689653488, 0.7236021006050218, 0.7211102550927978, 0.699714227381436, 0.7255342858886822, 0.581033561853358, 0.6144916598294886, 0.5571355310873648, 0.5219195340279955, 0.6954135460285485, 0.6029925372672534, 0.6390618123468182, 0.6681317235396027, 0.7280109889280518, 0.62, 0.5385164807134504, 0.6858571279792899, 0.62, 0.6660330322138686, 0.6462197768561405, 0.6289674077406556, 0.72773621594641, 0.64, 0.6954135460285485, 0.6696267617113282, 0.458257569495584, 0.6324555320336759, 0.6633249580710799, 0.6896375859826668, 0.7236021006050218, 0.5670978751503131, 0.47707441767506253, 0.54, 0.7211102550927978, 0.7255342858886822, 0.5399999999999999, 0.567097875150313, 0.7180529228406497, 0.656048778674269, 0.512249938994628, 0.5455272678794342, 0.6708203932499369, 0.6514598989960932, 0.7807688518377254, 0.7525955088890712, 0.6248199740725323, 0.7549834435270749, 0.530659966456864, 0.7507329751649385, 0.7255342858886822, 0.6082762530298219, 0.5571355310873648, 0.66, 0.6289674077406555, 0.6514598989960932, 0.7483314773547883, 0.7, 0.6633249580710799, 0.6633249580710799, 0.5385164807134504, 0.6324555320336759, 0.6514598989960932, 0.66, 0.5919459434779497]
bottom_end_count = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# # evenly sampled time at 200ms intervals
t = np.arange(0., len(bottom_mavg_reward), 1)

plt.subplot(3, 1, 1)
plt.plot(t, top_mavg_reward)
plt.plot(t, middle_mavg_reward)
plt.plot(t, bottom_mavg_reward)
plt.title('Windyworld Environment behavior per 1000 episodes')
plt.ylabel('Average reward')

plt.subplot(3, 1, 2)
plt.plot(t, top_mavg_steps)
plt.plot(t, middle_mavg_steps)
plt.plot(t, bottom_mavg_steps)
plt.ylabel('Average steps')

plt.subplot(3, 1, 3)
plt.plot(t, top_end_count, label='top')
plt.plot(t, middle_end_count, label='middle')
plt.plot(t, bottom_end_count, label='bottom')
plt.xlabel('Episodes x 50')
plt.ylabel('Average failure')

plt.legend()
plt.show()

