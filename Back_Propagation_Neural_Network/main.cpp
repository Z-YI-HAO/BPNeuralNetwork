#include<iostream>
#include<conio.h>
#include"net.h"
using namespace std;

int main() {
	int hidden_layer_num; //隐藏层的层数
	cout << "请输入隐藏层的层数：";
	cin >> hidden_layer_num;

	vector<int> hidden_layer_node_num(hidden_layer_num); //存放每一个隐藏层的结点的数量
	for (int i = 0; i < hidden_layer_num; i++) {
		cout << "请输入第" << i + 1 << "个隐藏层的结点数：";
		cin >> hidden_layer_node_num[i];
	}

	int max_epoch; //最大训练轮次
	cout << "请输入训练轮次：";
	cin >> max_epoch;

	Net net(hidden_layer_num, hidden_layer_node_num, max_epoch);

	net.Train("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	net.Predict("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

	cout << endl << "对测试集中前100张图片进行可视化展示和预测......";
	_getch();

	net.show_model("t10k-images.idx3-ubyte");

	cout << endl << endl << "对项目中最终的神经网络进行可视化展示......";
	_getch();
	cout << endl;

	net.show_net();
	net.output_net();

}