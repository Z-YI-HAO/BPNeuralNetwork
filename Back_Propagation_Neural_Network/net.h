#pragma once

#include<vector>
#include"config.h"
using namespace std;

//用于表示神经网络中的一个结点
class Node {
public:
	double value; //用于表示该结点的值
	double bias; //用于表示该结点的阈值
	vector<double> weight; //用于存放该结点接下去所有边的权值

	double bias_delta; //用于存放阈值的变化量
	vector<double> weight_delta; //用于存放该结点接下去所有权值的变化量

	explicit Node(int nextLayerSize); //用下一层的结点个数来对本层中存放权值的vector进行初始化
};

//用于对整个网络的模型进行表示
class Net {
private:
	Node* inputLayer[Config::INPUT_NODE_NUM]; //输入层的所有结点
	Node* outputLayer[Config::OUTPUT_NODE_NUM]; //输出层的所有结点

	int hidden_layer_num; //隐藏层的层数
	vector<int> hidden_layer_node_num; //存放每一个隐藏层的结点的数量
	Node*** hiddenLayer; //存放所有隐藏层的所有结点

	int max_epoch; //最大训练轮次

	void Forward(); //前向传播过程

	double CalcLoss(int* label); //损失函数

	void Backward(int* label); //后向传播过程

	void Update(int batch_size); //对网络中的权值和阈值进行更新

	int predict_label[100]; //存放测试集中前100个数据的预测结果，用于可视化展示

	pair<int, int> input_node_pos[3];
	pair<int, int> output_node_pos[10];
	pair<int, int>** hidden_node_pos;

public:
	Net(const int hidden_layer_num, const vector<int> hidden_layer_node_num, const int max_epoch);

	bool Train(const char* filename, const char* filelabel); //用训练集对网络进行训练

	void Predict(const char* filename, const char* filelabel); //用测试集对网络进行测试

	void show_model(const char* filename); //对预测结果进行可视化展示

	void cal_node_pos(int win_wid, int win_height);

	void show_net();

	void output_net();

	~Net();
};