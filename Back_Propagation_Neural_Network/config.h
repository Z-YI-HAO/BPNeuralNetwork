#pragma once

//此文件为项目的配置文件

namespace Config {
	const int INPUT_NODE_NUM = 784; //输入层的结点数
	const int OUTPUT_NODE_NUM = 10; //输出层的结点数
	const double LEARNING_RATE = 0.35; //学习率
	//const int MAX_EPOCH = 1; //最大训练轮次数
	const int BATCH_SIZE = 1000; //训练时每一批的数据个数

	const int node_radius = 22;
	const int node_gap = 60 + 14;
	const int layer_gap = 170 + 14;
	const int hidden_gap = 150 + 14;
	const int bound_gap = 40 + 14;
	const int node_delt = 30;
	const int input_node_gap = 90 + 14;
	const int input_node_delta = 10;
}