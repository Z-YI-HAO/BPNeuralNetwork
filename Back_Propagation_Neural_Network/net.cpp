#include<iostream>
#include<cmath>
#include<cstring>
#include<fstream>
#include<iomanip>
#include<graphics.h>
#include<conio.h>
#include"net.h"
using namespace std;

//生成-1到1之间的随机数
double rand_neg1_to_1()
{
	return (rand() % 1000 * 0.001 - 0.5) * 2;
}

//激活函数用到的sigmoid函数
double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

Node::Node(int nextLayerSize)
{
	//对存放各权值和各权值的变化量的vector进行初始化
	weight.resize(nextLayerSize);
	weight_delta.resize(nextLayerSize);

	value = 0;
	bias = 0;
	bias_delta = 0;
}

Net::Net(const int hidden_layer_num, const vector<int> hidden_layer_node_num, const int max_epoch)
{
	this->hidden_layer_num = hidden_layer_num;
	this->hidden_layer_node_num.assign(hidden_layer_node_num.begin(), hidden_layer_node_num.end());

	this->max_epoch = max_epoch;

	//为各个隐藏层申请空间
	hiddenLayer = new Node * *[hidden_layer_num];
	for (int i = 0; i < hidden_layer_num; i++) {
		hiddenLayer[i] = new Node * [hidden_layer_node_num[i]];
	}

	srand((unsigned)time(NULL));

	//根据第一个隐藏层的结点数对输入层结点进行初始化
	for (int i = 0; i < Config::INPUT_NODE_NUM; i++) {
		inputLayer[i] = new Node(hidden_layer_node_num[0]);

		for (int j = 0; j < hidden_layer_node_num[0]; j++) {
			inputLayer[i]->weight[j] = rand_neg1_to_1(); //产生-1到1之间的随机数，初始化权重
			inputLayer[i]->weight_delta[j] = 0; //初始化weight_delta
		}
	}

	//对各个隐藏层的结点进行初始化
	for (int layer = 0; layer < hidden_layer_num; layer++) {
		for (int i = 0; i < hidden_layer_node_num[layer]; i++) {
			if (layer != hidden_layer_num - 1) {
				hiddenLayer[layer][i] = new Node(hidden_layer_node_num[layer + 1]);

				for (int j = 0; j < hidden_layer_node_num[layer + 1]; j++) {
					hiddenLayer[layer][i]->weight[j] = rand_neg1_to_1();
					hiddenLayer[layer][i]->weight_delta[j] = 0;
				}
			}
			else {
				hiddenLayer[layer][i] = new Node(Config::OUTPUT_NODE_NUM);

				for (int j = 0; j < Config::OUTPUT_NODE_NUM; j++) {
					hiddenLayer[layer][i]->weight[j] = rand_neg1_to_1();
					hiddenLayer[layer][i]->weight_delta[j] = 0;
				}
			}
		}
	}

	//对输出层结点进行初始化
	for (int i = 0; i < Config::OUTPUT_NODE_NUM; i++) {
		outputLayer[i] = new Node(0);
		outputLayer[i]->bias = rand_neg1_to_1();
		outputLayer[i]->bias_delta = 0;
	}
}

void Net::Forward() {
	//输入层前向传递给隐藏层
	//公式为 $\alpha_j=\sigma(\sum_i {x_i*w_{ij}} -\beta_j)
	for (int i = 0; i < hidden_layer_node_num[0]; i++) {
		double sum = 0;
		for (int j = 0; j < Config::INPUT_NODE_NUM; j++) {
			sum += inputLayer[j]->value * inputLayer[j]->weight[i];
		}
		sum -= hiddenLayer[0][i]->bias;
		hiddenLayer[0][i]->value = sigmoid(sum);
	}

	//隐藏层之间依次前向传递
	//公式和输入层到隐藏层的公式相同
	for (int layer = 0; layer < hidden_layer_num - 1; layer++) {
		for (int i = 0; i < hidden_layer_node_num[layer + 1]; i++) {
			double sum = 0;
			for (int j = 0; j < hidden_layer_node_num[layer]; j++) {
				sum += hiddenLayer[layer][j]->value * hiddenLayer[layer][j]->weight[i];
			}
			sum -= hiddenLayer[layer + 1][i]->bias;
			hiddenLayer[layer + 1][i]->value = sigmoid(sum);
		}
	}

	//隐藏层前向传递给输出层
	for (int i = 0; i < Config::OUTPUT_NODE_NUM; i++) {
		double sum = 0;
		for (int j = 0; j < hidden_layer_node_num[hidden_layer_num - 1]; j++) {
			sum += hiddenLayer[hidden_layer_num - 1][j]->value * hiddenLayer[hidden_layer_num - 1][j]->weight[i];
		}
		sum -= outputLayer[i]->bias;
		outputLayer[i]->value = sigmoid(sum);
	}
}

//损失函数
double Net::CalcLoss(int* label)
{
	double loss = 0;

	//损失函数的公式：loss = \frac{1}{2}\sum_k(y_k-\hat{y_k})^2
	for (int i = 0; i < Config::OUTPUT_NODE_NUM; i++) {
		double temp = fabs(outputLayer[i]->value - label[i]);
		loss += temp * temp / 2;
	}
	return loss;
}

//后向传播过程
void Net::Backward(int* label)
{
	//由于每一层阈值的变化量和前一层到该层的权重的变化量的公式只差一个前一层结点的值
	//所以计算时只需计算当前层阈值的变化量

	//计算输出层各个结点的bias_delta，用于更新其bias
	//计算公式为 \delta_k^K=(o_k-t_k)o_k(1-o_k)
	for (int k = 0; k < Config::OUTPUT_NODE_NUM; k++) {
		outputLayer[k]->bias_delta = (outputLayer[k]->value) * (1.0 - outputLayer[k]->value) * (outputLayer[k]->value - label[k]);
	}

	//计算所有隐藏层的各个结点的bias_delta，用于更新其bias
	//计算公式为 \delta_j^J=o_j(1-o_j)·\sum_k \delta_k^K·w_{jk}
	for (int layer = hidden_layer_num - 1; layer >= 0; layer--) {
		for (int j = 0; j < hidden_layer_node_num[layer]; j++) {
			double sigma = 0;
			if (layer == hidden_layer_num - 1) {
				for (int k = 0; k < Config::OUTPUT_NODE_NUM; k++) {
					sigma += hiddenLayer[layer][j]->weight[k] * outputLayer[k]->bias_delta;
				}
				hiddenLayer[layer][j]->bias_delta = (hiddenLayer[layer][j]->value) * (1.0 - hiddenLayer[layer][j]->value) * sigma;
			}
			else {
				for (int k = 0; k < hidden_layer_node_num[layer + 1]; k++) {
					sigma += hiddenLayer[layer][j]->weight[k] * hiddenLayer[layer + 1][k]->bias_delta;
				}
				hiddenLayer[layer][j]->bias_delta = (hiddenLayer[layer][j]->value) * (1.0 - hiddenLayer[layer][j]->value) * sigma;
			}
		}
	}
}

//对网络中的权值和阈值进行更新
void Net::Update(int batch_size) {
	for (int layer = 0; layer < hidden_layer_num; layer++) {
		for (int j = 0; j < hidden_layer_node_num[layer]; j++) {
			//用weigh_delta对该隐藏层各结点的阈值进行更新
			hiddenLayer[layer][j]->bias = hiddenLayer[layer][j]->bias - Config::LEARNING_RATE * hiddenLayer[layer][j]->bias_delta;

			if (layer == 0) {
				//对输入层到隐藏层的各个权重进行更新
				for (int i = 0; i < Config::INPUT_NODE_NUM; i++) {
					inputLayer[i]->weight[j] = inputLayer[i]->weight[j] - Config::LEARNING_RATE * inputLayer[i]->value * hiddenLayer[layer][j]->bias_delta;
				}
			}
			else {
				//对隐藏层之间的各个权重进行更新
				for (int i = 0; i < hidden_layer_node_num[layer - 1]; i++) {
					hiddenLayer[layer - 1][i]->weight[j] = hiddenLayer[layer - 1][i]->weight[j] - Config::LEARNING_RATE * hiddenLayer[layer - 1][i]->value * hiddenLayer[layer][j]->bias_delta;
				}
			}
		}
	}

	for (int k = 0; k < Config::OUTPUT_NODE_NUM; k++) {
		//对输出层各结点的阈值进行更新
		outputLayer[k]->bias = outputLayer[k]->bias - Config::LEARNING_RATE * outputLayer[k]->bias_delta;

		//对最后一个隐藏层到输出层的各个权重进行更新
		for (int j = 0; j < hidden_layer_node_num[hidden_layer_num - 1]; j++) {
			hiddenLayer[hidden_layer_num - 1][j]->weight[k] = hiddenLayer[hidden_layer_num - 1][j]->weight[k] - Config::LEARNING_RATE * hiddenLayer[hidden_layer_num - 1][j]->value * outputLayer[k]->bias_delta;
		}
	}
}

bool Net::Train(const char* filename, const char* filelabel)
{
	ifstream dataFile(filename, ios::in | ios::binary);
	ifstream labelFile(filelabel, ios::in | ios::binary);

	if (!dataFile.is_open()) {
		cout << "Error in file opening, file path: " << filename << endl;
		return false;
	}
	if (!labelFile.is_open()) {
		cout << "Error in file opening, file path: " << filelabel << endl;
		return false;
	}

	char image_buf[784]; //用于存放一张图片的数据
	int label_buf[10]; //用于存放一张图片的标签（与读入的标签相对应）

	char label_temp; //用于存放读入的标签

	//进行多轮训练
	for (int epoch = 0; epoch < max_epoch; epoch++) {
		char useless_data[16]; //存放数据文件中没有用的字节
		char useless_label[8]; //存放标签文件中没有用的字节

		dataFile.clear();
		labelFile.clear();

		dataFile.seekg(0, ios::beg);
		labelFile.seekg(0, ios::beg);

		dataFile.read(useless_data, 16);
		labelFile.read(useless_label, 8);

		double total_loss = 0;
		double average_loss = 0;
		int cnt = 0;

		//ResetZero(); //将现有的累计量清零

		cout << endl << "#epoch " << epoch + 1 << " Training peocess: " << endl;

		while (dataFile.peek() != EOF && labelFile.peek() != EOF) {

			total_loss = 0;

			for (int j = 0; j < Config::BATCH_SIZE; j++) {
				memset(image_buf, 0, 784);
				memset(label_buf, 0, 10);

				dataFile.read(image_buf, 784);
				labelFile.read(&label_temp, 1);

				for (int i = 0; i < 10; i++) {
					label_buf[i] = (unsigned int)label_temp == i ? 1 : 0;
				}

				for (int i = 0; i < Config::INPUT_NODE_NUM; i++) {
					inputLayer[i]->value = (unsigned int)image_buf[i] < 128 ? 0 : 1;
					//inputLayer[i]->value = image_buf[i] / 255;
				}

				Forward();

				total_loss += CalcLoss(label_buf);

				Backward(label_buf);

				Update(Config::BATCH_SIZE);
			}

			average_loss = total_loss / Config::BATCH_SIZE;
			cout << "cnt: " << setw(2) << setfill('0') << ++cnt * 1000 << " loss:" << average_loss << endl;
		}

		cout << "#epoch " << epoch + 1 << " --average_loss: " << average_loss << endl;
	}

	dataFile.close();
	labelFile.close();

	return true;
}

void Net::Predict(const char* filename, const char* filelabel)
{
	ifstream dataFile(filename, ios::in | ios::binary);
	ifstream labelFile(filelabel, ios::in | ios::binary);

	if (!dataFile.is_open()) {
		cout << "Error in file opening, file path: " << filename << endl;
		return;
	}
	if (!labelFile.is_open()) {
		cout << "Error in file opening, file path: " << filelabel << endl;
		return;
	}

	int success_num = 0; //预测成功的结点
	int test_num = 0; //测试的数量

	int count = 0;

	char image_buf[784];
	int label_buf[10];

	char label_temp;

	char useless_data[16]; //存放数据文件中没有用的字节
	char useless_label[8]; //存放标签文件中没有用的字节
	dataFile.read(useless_data, 16);
	labelFile.read(useless_label, 8);

	cout << endl << "Predict result: " << endl;

	while (dataFile.peek() != EOF && labelFile.peek() != EOF) {
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);

		dataFile.read(image_buf, 784);
		labelFile.read(&label_temp, 1);

		for (int i = 0; i < 10; i++) {
			label_buf[i] = label_temp == (unsigned int)i ? 1 : 0;
		}

		for (int i = 0; i < Config::INPUT_NODE_NUM; i++) {
			inputLayer[i]->value = (unsigned int)image_buf[i] < 128 ? 0 : 1;
			//inputLayer[i]->value = image_buf[i] / 255;
		}

		Forward();

		//用于记录输出层中值最大的位置
		double max_value = -99999;
		int max_index = 0;

		for (int i = 0; i < Config::OUTPUT_NODE_NUM; i++) {
			if (outputLayer[i]->value > max_value) {
				max_value = outputLayer[i]->value;
				max_index = i;
			}
		}

		if (count < 100) {
			this->predict_label[count] = max_index;
			count++;
		}

		if (label_buf[max_index] == 1) {
			success_num++;
		}

		test_num++;

		if (test_num % 1000 == 0) {
			cout << "Test num: " << test_num << " success : " << success_num << endl;
		}
	}

	cout << endl;
	cout << "Success rate: " << double(success_num * 1.0 / test_num) << endl; //计算预测的准确率并将其输出
}

Net::~Net()
{
	for (int i = 0; i < Config::INPUT_NODE_NUM; i++) {
		delete inputLayer[i];
	}

	for (int i = 0; i < Config::OUTPUT_NODE_NUM; i++) {
		delete outputLayer[i];
	}

	for (int i = 0; i < hidden_layer_num; i++) {
		for (int j = 0; j < hidden_layer_node_num[i]; j++) {
			delete hiddenLayer[i][j];
		}
		delete[] hiddenLayer[i];
	}
	delete[] hiddenLayer;
}

//用于画一张图片及其标签
void draw_pic(char mat[784], int pred, int row, int col) {
	int root_x = col * 28 * 2;
	int root_y = row * 28 * 2 + 28;
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (i == 0 || i == 27 || j == 0 || j == 27 || mat[i * 28 + j] == char(1))
				putpixel(root_x + j + 50, root_y + i + 50, RED);
		}
	}

	setcolor(BLACK);
	settextstyle(25, 18, L"黑体");
	outtextxy(root_x + 55, root_y - 28 + 50, '0' + pred);
}

void Net::show_model(const char* filename) {
	ifstream indata(filename, ios::in | ios::binary);
	if (!indata.is_open()) {
		cout << "Error in opening file!" << endl;
		return;
	}

	char useless[16];
	indata.read(useless, 16);

	initgraph(632, 682);
	setbkcolor(WHITE);
	cleardevice();

	setcolor(BLACK);
	outtextxy(10, 10, L"前100张图片的像素点阵输出以及其预测值");
	outtextxy(10, 650, L"按任意键退出...");

	settextstyle(20, 20, L"黑体");
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			unsigned char buff[784];
			memset(buff, 0, 784);
			indata.read((char*)(buff), 784);

			char tran[784];
			memset(tran, 0, 784);
			for (int k = 0; k < 784; k++) {
				tran[k] = (unsigned int)(buff[k]) < 128 ? char(0) : char(1);
			}
			draw_pic(tran, predict_label[i * 10 + j], i, j);
		}
	}
	indata.close();

	_getch();
	closegraph();
}

void Net::cal_node_pos(int win_wid, int win_height) {
	int win_mid = win_height / 2;
	//计算输入层节点坐标
	input_node_pos[0] = make_pair(Config::bound_gap, win_mid - Config::input_node_gap);
	input_node_pos[1] = make_pair(Config::bound_gap, win_mid);
	input_node_pos[2] = make_pair(Config::bound_gap, win_mid + Config::input_node_gap);
	//计算隐藏层节点坐标
	//申先申请隐藏层节点坐标的存储空间
	hidden_node_pos = new pair<int, int>*[hidden_layer_num];
	for (int i = 0; i < hidden_layer_num; i++)
		hidden_node_pos[i] = new pair<int, int>[hidden_layer_node_num[i]];

	for (int i = 0; i < hidden_layer_num; i++) {
		if (hidden_layer_node_num[i] <= 10) {
			bool odd_flag = hidden_layer_node_num[i] % 2;
			for (int j = 0; j < hidden_layer_node_num[i]; j++) {
				int x = Config::bound_gap + Config::layer_gap + i * Config::hidden_gap;
				int y;
				if (odd_flag) {
					y = win_mid + (j - hidden_layer_node_num[i] / 2) * Config::node_gap;
				}
				else {
					double mid = (hidden_layer_node_num[i] - 1) * 1.0 / 2;
					y = (int)(win_mid + (j - mid) * Config::node_gap);
				}
				hidden_node_pos[i][j] = make_pair(x, y);
			}
		}
		else {
			bool odd_flag = 10 % 2;
			for (int j = 0; j < 10; j++) {
				int x = Config::bound_gap + Config::layer_gap + i * Config::hidden_gap;
				int y;
				if (odd_flag) {
					y = win_mid + (j - 10 / 2) * Config::node_gap;
				}
				else {
					double mid = (10 - 1) * 1.0 / 2;
					y = (int)(win_mid + (j - mid) * Config::node_gap);
				}
				hidden_node_pos[i][j] = make_pair(x, y);
			}
		}
	}

	//计算输出层节点坐标
	for (int i = 0; i < 5; i++) {
		output_node_pos[4 - i] = make_pair(win_wid - Config::bound_gap, win_mid - Config::node_gap / 2 - Config::node_gap * i);
		output_node_pos[5 + i] = make_pair(win_wid - Config::bound_gap, win_mid + Config::node_gap / 2 + Config::node_gap * i);
	}
}

void Net::show_net() {
	int win_wid, win_height, win_mid;
	int max_node = 0;
	int max_layer = 2 + hidden_layer_num;
	for (int i = 0; i < hidden_layer_num; i++) {
		if (hidden_layer_node_num[i] > max_node)
			max_node = hidden_layer_node_num[i];
	}

	max_node = max_node <= 10 ? max_node : 10; //图上最多画10个结点

	win_wid = 2 * Config::layer_gap + (hidden_layer_num - 1) * Config::hidden_gap + Config::bound_gap * 2;
	win_height = 2 * Config::bound_gap + (max_node - 1) * Config::node_gap;
	win_mid = win_height / 2;

	cal_node_pos(win_wid, win_height);

	initgraph(win_wid, win_height);
	setbkcolor(WHITE);
	cleardevice();
	//画输入层节点
	setfillcolor(COLORREF(0x2679f4));//先设置填充颜色#f47926
	//全部填充
	fillcircle(input_node_pos[0].first, input_node_pos[0].second, Config::node_radius);
	fillcircle(input_node_pos[1].first, input_node_pos[1].second, Config::node_radius);
	fillcircle(input_node_pos[2].first, input_node_pos[2].second, Config::node_radius);

	//画隐藏层节点
	setfillcolor(COLORREF(0x59b67c));
	//全部填充
	for (int i = 0; i < hidden_layer_num; i++) {
		//如果当前隐藏层结点数不超过10，则全部进行显示
		if (hidden_layer_node_num[i] <= 10) {
			for (int j = 0; j < hidden_layer_node_num[i]; j++) {
				fillcircle(hidden_node_pos[i][j].first, hidden_node_pos[i][j].second, Config::node_radius);
			}
		}
		else {
			for (int j = 0; j < 10; j++) {
				fillcircle(hidden_node_pos[i][j].first, hidden_node_pos[i][j].second, Config::node_radius);
			}
		}
	}

	//画输出层节点
	setfillcolor(COLORREF(0xc96f3d));
	//全部填充
	for (int i = 0; i < 10; i++)
		fillcircle(output_node_pos[i].first, output_node_pos[i].second, Config::node_radius);


	setlinecolor(BLACK);//参数可以是颜色，也可以是三原色
	setlinestyle(PS_SOLID, 1);//参数linestyle可以点进去库函数查看，可以设置虚线、直线....,width是线的宽度
	//画输入层到隐藏层节点的连线
	for (int i = 0; i < 3; i++) {
		//如果第一个隐藏层结点数不超过10，则全部进行连线
		if (hidden_layer_node_num[0] <= 10) {
			for (int j = 0; j < hidden_layer_node_num[0]; j++) {
				line(input_node_pos[i].first + Config::node_radius, input_node_pos[i].second,
					hidden_node_pos[0][j].first - Config::node_radius, hidden_node_pos[0][j].second);
			}
		}
		else {
			for (int j = 0; j < 10; j++) {
				line(input_node_pos[i].first + Config::node_radius, input_node_pos[i].second,
					hidden_node_pos[0][j].first - Config::node_radius, hidden_node_pos[0][j].second);
			}
		}
	}

	//画隐藏层之间的连线
	for (int i = 1; i < hidden_layer_num; i++) {
		if (hidden_layer_node_num[i] <= 10) {
			for (int j = 0; j < hidden_layer_node_num[i]; j++) {
				//如果当前结点的个数小于等于10，则当前层的最后一个结点和下一层所有结点连接
				if (hidden_layer_node_num[i - 1] <= 10) {
					line(hidden_node_pos[i - 1][0].first + Config::node_radius, hidden_node_pos[i - 1][0].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
					line(hidden_node_pos[i - 1][hidden_layer_node_num[i - 1] - 1].first + Config::node_radius, hidden_node_pos[i - 1][hidden_layer_node_num[i - 1] - 1].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
				}
				else {
					line(hidden_node_pos[i - 1][0].first + Config::node_radius, hidden_node_pos[i - 1][0].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
					line(hidden_node_pos[i - 1][10 - 1].first + Config::node_radius, hidden_node_pos[i - 1][10 - 1].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
				}
			}
		}
		else {
			for (int j = 0; j < 10; j++) {
				//如果当前结点的个数小于等于10，则当前层的最后一个结点和下一层所有结点连接
				if (hidden_layer_node_num[i - 1] <= 10) {
					line(hidden_node_pos[i - 1][0].first + Config::node_radius, hidden_node_pos[i - 1][0].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
					line(hidden_node_pos[i - 1][hidden_layer_node_num[i - 1] - 1].first + Config::node_radius, hidden_node_pos[i - 1][hidden_layer_node_num[i - 1] - 1].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
				}
				else {
					line(hidden_node_pos[i - 1][0].first + Config::node_radius, hidden_node_pos[i - 1][0].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
					line(hidden_node_pos[i - 1][10 - 1].first + Config::node_radius, hidden_node_pos[i - 1][10 - 1].second,
						hidden_node_pos[i][j].first - Config::node_radius, hidden_node_pos[i][j].second);
				}
			}
		}
	}

	//画隐藏层之间的虚线
	setlinecolor(COLORREF(0x0000FF));//参数可以是颜色，也可以是三原色
	setlinestyle(PS_DOT, 5);//参数linestyle可以点进去库函数查看，可以设置虚线、直线....,width是线的宽度
	for (int i = 0; i < hidden_layer_num - 1; i++) {
		if (hidden_layer_node_num[i] <= 10) {
			for (int j = 1; j < hidden_layer_node_num[i] - 1; j++) {
				line(hidden_node_pos[i][j].first + Config::node_radius + Config::node_delt, hidden_node_pos[i][j].second,
					hidden_node_pos[i][j].first + Config::hidden_gap - Config::node_delt - Config::node_radius, hidden_node_pos[i][j].second);
			}
		}
		else {
			for (int j = 1; j < 10 - 1; j++) {
				line(hidden_node_pos[i][j].first + Config::node_radius + Config::node_delt, hidden_node_pos[i][j].second,
					hidden_node_pos[i][j].first + Config::hidden_gap - Config::node_delt - Config::node_radius, hidden_node_pos[i][j].second);
			}

			//画当前隐藏层中间位置的虚线
			line(hidden_node_pos[i][4].first, hidden_node_pos[i][4].second + Config::node_radius + Config::node_gap / 15,
				hidden_node_pos[i][4].first, hidden_node_pos[i][5].second - Config::node_radius);
		}
	}

	setlinecolor(BLACK);//参数可以是颜色，也可以是三原色
	setlinestyle(PS_SOLID, 1);//参数linestyle可以点进去库函数查看，可以设置虚线、直线....,width是线的宽度
	/*
	for (int i = 0; i < hidden_layer_num-1; i++) {
		for (int j = 0; j < hidden_layer_node_num[i]; j++) {
			for (int k = 0; k < hidden_layer_node_num[i + 1]; k++) {
				line(hidden_node_pos[i][j].first + Config::node_radius, hidden_node_pos[i][j].second,
					hidden_node_pos[i + 1][k].first - Config::node_radius, hidden_node_pos[i + 1][k].second);
			}
		}
	}
	*/
	//画隐藏层与输出层之间的连线
	for (int j = 0; j < Config::OUTPUT_NODE_NUM; j++) {
		if (hidden_layer_node_num[hidden_layer_num - 1] <= 10) {
			line(hidden_node_pos[hidden_layer_num - 1][0].first + Config::node_radius, hidden_node_pos[hidden_layer_num - 1][0].second,
				output_node_pos[j].first - Config::node_radius, output_node_pos[j].second);
			line(hidden_node_pos[hidden_layer_num - 1][hidden_layer_node_num[hidden_layer_num - 1] - 1].first + Config::node_radius, hidden_node_pos[hidden_layer_num - 1][hidden_layer_node_num[hidden_layer_num - 1] - 1].second,
				output_node_pos[j].first - Config::node_radius, output_node_pos[j].second);
		}
		else {
			line(hidden_node_pos[hidden_layer_num - 1][0].first + Config::node_radius, hidden_node_pos[hidden_layer_num - 1][0].second,
				output_node_pos[j].first - Config::node_radius, output_node_pos[j].second);
			line(hidden_node_pos[hidden_layer_num - 1][10 - 1].first + Config::node_radius, hidden_node_pos[hidden_layer_num - 1][10 - 1].second,
				output_node_pos[j].first - Config::node_radius, output_node_pos[j].second);
		}
	}
	/*
	for (int i = 0; i < hidden_layer_node_num[hidden_layer_num - 1]; i++) {
		for (int j = 0; j < Config::OUTPUT_NODE_NUM; j++) {
			line(hidden_node_pos[hidden_layer_num - 1][i].first + Config::node_radius, hidden_node_pos[hidden_layer_num - 1][i].second,
				output_node_pos[j].first - Config::node_radius, output_node_pos[j].second);
		}
	}
	*/
	//画隐藏层与输出层之间的虚线
	setlinecolor(COLORREF(0x0000ff));//参数可以是颜色，也可以是三原色
	setlinestyle(PS_DOT, 5);//参数linestyle可以点进去库函数查看，可以设置虚线、直线....,width是线的宽度
	if (hidden_layer_node_num[hidden_layer_num - 1] <= 10) {
		for (int i = 1; i < hidden_layer_node_num[hidden_layer_num - 1] - 1; i++) {
			line(hidden_node_pos[hidden_layer_num - 1][i].first + Config::node_radius + Config::node_delt, hidden_node_pos[hidden_layer_num - 1][i].second,
				hidden_node_pos[hidden_layer_num - 1][i].first + Config::layer_gap - Config::node_delt - Config::node_radius, hidden_node_pos[hidden_layer_num - 1][i].second);
		}
	}
	else {
		for (int i = 1; i < 10 - 1; i++) {
			line(hidden_node_pos[hidden_layer_num - 1][i].first + Config::node_radius + Config::node_delt, hidden_node_pos[hidden_layer_num - 1][i].second,
				hidden_node_pos[hidden_layer_num - 1][i].first + Config::layer_gap - Config::node_delt - Config::node_radius, hidden_node_pos[hidden_layer_num - 1][i].second);
		}

		//画最后一个隐藏层中间位置的虚线
		line(hidden_node_pos[hidden_layer_num - 1][4].first, hidden_node_pos[hidden_layer_num - 1][4].second + Config::node_radius + Config::node_gap / 15,
			hidden_node_pos[hidden_layer_num - 1][4].first, hidden_node_pos[hidden_layer_num - 1][5].second - Config::node_radius);
	}

	//画输入层节点之间的虚线
	line(input_node_pos[0].first, input_node_pos[0].second + Config::node_radius + Config::input_node_delta, input_node_pos[1].first, input_node_pos[1].second - Config::node_radius - Config::input_node_delta);
	line(input_node_pos[1].first, input_node_pos[1].second + Config::node_radius + Config::input_node_delta, input_node_pos[2].first, input_node_pos[2].second - Config::node_radius);

	//标注输入层，隐藏层，输出层
	setcolor(BLACK);//文字颜色
	settextstyle(20, 10, L"黑体");
	setbkmode(TRANSPARENT);//文字背景透明
	outtextxy(input_node_pos[2].first - 55, input_node_pos[2].second + Config::node_radius + 30, L"Input Layer");
	outtextxy(output_node_pos[Config::OUTPUT_NODE_NUM - 1].first - 100, output_node_pos[Config::OUTPUT_NODE_NUM - 1].second + Config::node_radius, L"Output Layer");
	int max_hidden_node = 0, max_index = 0;
	for (int i = 0; i < hidden_layer_num; i++) {
		if (max_hidden_node < hidden_layer_node_num[i]) {
			max_hidden_node = hidden_layer_node_num[i];
			max_index = i;
		}
	}
	for (int i = 0; i < hidden_layer_num; i++) {
		string temp = "Hidden Layer ";
		temp = temp + char(i + 1 + '0');
		size_t size = temp.length();
		wchar_t* buffer = new wchar_t[size + 1];
		MultiByteToWideChar(CP_ACP, 0, temp.c_str(), size, buffer, size * sizeof(wchar_t));
		buffer[size] = 0;
		if (hidden_layer_node_num[max_index] <= 10) {
			outtextxy(hidden_node_pos[i][0].first - 70, hidden_node_pos[max_index][hidden_layer_node_num[max_index] - 1].second + Config::node_radius, buffer);
		}
		else {
			outtextxy(hidden_node_pos[i][0].first - 70, hidden_node_pos[max_index][10 - 1].second + Config::node_radius, buffer);
		}
		delete[] buffer;
	}

	setcolor(WHITE);//文字颜色
	settextstyle(20, 8, L"黑体");
	//画隐藏层的 Σ|σ
	for (int i = 0; i < hidden_layer_num; i++) {
		for (int j = 0; j < hidden_layer_node_num[i]; j++) {
			outtextxy(hidden_node_pos[i][j].first - Config::node_radius * 3 / 4, hidden_node_pos[i][j].second - Config::node_radius / 2, L"Σ|σ");
		}
	}
	//画输出层的 Σ|σ
	for (int i = 0; i < Config::OUTPUT_NODE_NUM; i++) {
		outtextxy(output_node_pos[i].first - Config::node_radius * 3 / 4, output_node_pos[i].second - Config::node_radius / 2, L"Σ|σ");
	}

	//为输入层节点标号
	setcolor(COLORREF(0x696969));//文字颜色
	settextstyle(20, 8, L"黑体");
	outtextxy(input_node_pos[0].first - Config::node_radius - 20, input_node_pos[0].second - 10, L"X1");
	outtextxy(input_node_pos[1].first - Config::node_radius - 20, input_node_pos[1].second - 10, L"Xi");
	outtextxy(input_node_pos[2].first - 54, input_node_pos[2].second - 10, L"X783");

	//标记隐藏层节点上方的BIAS
	setcolor(COLORREF(0x59b67c));
	settextstyle(20, 10, L"黑体");
	for (int i = 0; i < hidden_layer_num; i++) {
		outtextxy(hidden_node_pos[i][0].first - Config::node_radius * 2, hidden_node_pos[i][0].second - Config::node_radius * 2, L"Bias(1-k)");
	}
	//标记输出层节点上方的BIAS
	setcolor(COLORREF(0xc96f3d));
	settextstyle(20, 10, L"黑体");
	outtextxy(output_node_pos[0].first - Config::node_radius * 2, output_node_pos[0].second - Config::node_radius * 2, L"Bias(1-k)");

	//标注输入层到隐藏层的w
	setlinecolor(COLORREF(0x1E69D2));//参数可以是颜色，也可以是三原色
	setlinestyle(PS_DASH | PS_ENDCAP_SQUARE, 3);
	line(input_node_pos[0].first + Config::node_radius, (input_node_pos[0].second + hidden_node_pos[0][0].second) / 2 - Config::node_radius * 3 + 10,
		hidden_node_pos[0][0].first - Config::node_radius, (input_node_pos[0].second + hidden_node_pos[0][0].second) / 2 - Config::node_radius * 3 + 10);
	setcolor(COLORREF(0x1E69D2));
	settextstyle(20, 10, L"黑体");
	setbkmode(OPAQUE);//文字背景透明
	outtextxy((input_node_pos[0].first + hidden_node_pos[0][0].first) / 2,
		(input_node_pos[0].second + hidden_node_pos[0][0].second) / 2 - Config::node_radius * 3, L"Wij");

	//标注隐藏层到隐藏层的w
	for (int i = 0; i < hidden_layer_num - 1; i++) {
		line(hidden_node_pos[i][0].first + Config::node_radius, (hidden_node_pos[i][0].second + hidden_node_pos[i + 1][0].second) / 2 - Config::node_radius * 4 + 10,
			hidden_node_pos[i + 1][0].first - Config::node_radius, (hidden_node_pos[i][0].second + hidden_node_pos[i + 1][0].second) / 2 - Config::node_radius * 4 + 10);
		outtextxy((hidden_node_pos[i][0].first + hidden_node_pos[i + 1][0].first) / 2,
			(hidden_node_pos[i][0].second + hidden_node_pos[i + 1][0].second) / 2 - Config::node_radius * 4, L"Wij");
	}
	//标注隐藏层到输出层的w
	line(hidden_node_pos[hidden_layer_num - 1][0].first + Config::node_radius, output_node_pos[0].second + 10 - Config::node_radius,
		output_node_pos[0].first - Config::node_radius, output_node_pos[0].second + 10 - Config::node_radius);

	outtextxy((hidden_node_pos[hidden_layer_num - 1][0].first + output_node_pos[0].first) / 2,
		output_node_pos[0].second - Config::node_radius, L"Wij");


	_getch();
	closegraph();

}

void Net::output_net() {
	while (1) {
		cout << endl << "您可以通过输入来选择各节点之间数据传递的权重，或者某节点的偏置值" << endl;
		cout << "您是否要查看？（输入 1/0 表示 是/否）: ";
		while (1) {
			char temp = getchar();
			if (temp == '0')
				return;
			if (temp != '1') {
				if (temp != '\n')
					cout << "输入错误" << endl;
				continue;
			}
			break;
		}
		cout << endl << "您要查看权重/偏置值？（输入 1/0 表示 权重/偏置值）";
		int which = 0;
		while (1) {
			char temp = getchar();
			if (temp == '0') {
				which = 0;
				break;
			}
			else if (temp == '1') {
				which = 1;
				break;
			}
			else if (temp == '\n') {
				continue;
			}
			else
				cout << "输入错误" << endl;
		}
		if (which) {
			cout << endl << "您要查看节点之间的权重，将输入层看作第0层，输出层看作最后一层，请输入要查看的层(该层和该层的下一层即为要选择的两层):";
			int layer;
			cin >> layer;
			cout << "请输入要查看的两个层数上节点的位置:";
			int pos1, pos2;
			cin >> pos1 >> pos2;
			cout << "权值:";
			if (layer == 0)
				cout << inputLayer[pos1]->weight[pos2] << endl;
			else if (layer == 2 + hidden_layer_num - 2)
				cout << hiddenLayer[hidden_layer_num - 1][pos1]->weight[pos2] << endl;
			else
				cout << hiddenLayer[layer - 1][pos1]->weight[pos2] << endl;
		}
		else {
			cout << endl << "您要查看节点的偏置值，将输入层看作第0层，输出层看作最后一层，请输入要查看节点所在的层:";
			int layer;
			cin >> layer;
			cout << "请输入您要查看的节点标号:";
			int pos;
			cin >> pos;
			cout << "偏置值:";
			if (layer == layer == 2 + hidden_layer_num - 2)
				cout << outputLayer[pos]->bias << endl;
			else
				cout << hiddenLayer[layer - 1][pos]->bias << endl;
		}

	}
}


