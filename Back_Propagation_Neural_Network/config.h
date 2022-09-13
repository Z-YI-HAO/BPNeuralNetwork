#pragma once

//���ļ�Ϊ��Ŀ�������ļ�

namespace Config {
	const int INPUT_NODE_NUM = 784; //�����Ľ����
	const int OUTPUT_NODE_NUM = 10; //�����Ľ����
	const double LEARNING_RATE = 0.35; //ѧϰ��
	//const int MAX_EPOCH = 1; //���ѵ���ִ���
	const int BATCH_SIZE = 1000; //ѵ��ʱÿһ�������ݸ���

	const int node_radius = 22;
	const int node_gap = 60 + 14;
	const int layer_gap = 170 + 14;
	const int hidden_gap = 150 + 14;
	const int bound_gap = 40 + 14;
	const int node_delt = 30;
	const int input_node_gap = 90 + 14;
	const int input_node_delta = 10;
}