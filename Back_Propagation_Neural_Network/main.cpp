#include<iostream>
#include<conio.h>
#include"net.h"
using namespace std;

int main() {
	int hidden_layer_num; //���ز�Ĳ���
	cout << "���������ز�Ĳ�����";
	cin >> hidden_layer_num;

	vector<int> hidden_layer_node_num(hidden_layer_num); //���ÿһ�����ز�Ľ�������
	for (int i = 0; i < hidden_layer_num; i++) {
		cout << "�������" << i + 1 << "�����ز�Ľ������";
		cin >> hidden_layer_node_num[i];
	}

	int max_epoch; //���ѵ���ִ�
	cout << "������ѵ���ִΣ�";
	cin >> max_epoch;

	Net net(hidden_layer_num, hidden_layer_node_num, max_epoch);

	net.Train("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	net.Predict("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

	cout << endl << "�Բ��Լ���ǰ100��ͼƬ���п��ӻ�չʾ��Ԥ��......";
	_getch();

	net.show_model("t10k-images.idx3-ubyte");

	cout << endl << endl << "����Ŀ�����յ���������п��ӻ�չʾ......";
	_getch();
	cout << endl;

	net.show_net();
	net.output_net();

}