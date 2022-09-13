#pragma once

#include<vector>
#include"config.h"
using namespace std;

//���ڱ�ʾ�������е�һ�����
class Node {
public:
	double value; //���ڱ�ʾ�ý���ֵ
	double bias; //���ڱ�ʾ�ý�����ֵ
	vector<double> weight; //���ڴ�Ÿý�����ȥ���бߵ�Ȩֵ

	double bias_delta; //���ڴ����ֵ�ı仯��
	vector<double> weight_delta; //���ڴ�Ÿý�����ȥ����Ȩֵ�ı仯��

	explicit Node(int nextLayerSize); //����һ��Ľ��������Ա����д��Ȩֵ��vector���г�ʼ��
};

//���ڶ����������ģ�ͽ��б�ʾ
class Net {
private:
	Node* inputLayer[Config::INPUT_NODE_NUM]; //���������н��
	Node* outputLayer[Config::OUTPUT_NODE_NUM]; //���������н��

	int hidden_layer_num; //���ز�Ĳ���
	vector<int> hidden_layer_node_num; //���ÿһ�����ز�Ľ�������
	Node*** hiddenLayer; //����������ز�����н��

	int max_epoch; //���ѵ���ִ�

	void Forward(); //ǰ�򴫲�����

	double CalcLoss(int* label); //��ʧ����

	void Backward(int* label); //���򴫲�����

	void Update(int batch_size); //�������е�Ȩֵ����ֵ���и���

	int predict_label[100]; //��Ų��Լ���ǰ100�����ݵ�Ԥ���������ڿ��ӻ�չʾ

	pair<int, int> input_node_pos[3];
	pair<int, int> output_node_pos[10];
	pair<int, int>** hidden_node_pos;

public:
	Net(const int hidden_layer_num, const vector<int> hidden_layer_node_num, const int max_epoch);

	bool Train(const char* filename, const char* filelabel); //��ѵ�������������ѵ��

	void Predict(const char* filename, const char* filelabel); //�ò��Լ���������в���

	void show_model(const char* filename); //��Ԥ�������п��ӻ�չʾ

	void cal_node_pos(int win_wid, int win_height);

	void show_net();

	void output_net();

	~Net();
};