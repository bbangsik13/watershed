#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <cmath>
#include <queue>
#include <chrono>
#include <random>
#include <list>
#include <vector>


struct WS_struct
{
public:
    float dist;
    int x;
    int y;
    int marker;


    WS_struct(float dist, int x, int y, int marker)
    {
        this->dist = dist;
        this->x = x;
        this->y = y;
        this->marker = marker;
    }
};


bool operator<(const WS_struct& a, const WS_struct& b)
{
    return a.dist > b.dist;
}


void add_neighbor(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img, cv::Mat tag) {
    if (x - 1 >= 0) {//add left
        if (!tag.at<uchar>(x - 1, y)) {//tag 안 되어 있으면 queue 넣기 (dist, x, y, marker)
            WS_struct wst(abs(img.at<uchar>(x, y) - img.at<uchar>(x - 1, y)), x - 1, y, marker);
            WS_que.push(wst);
        }
    }

    if (x + 1 < img.cols) {//add right
        if (!tag.at<uchar>(x + 1, y)) {
            WS_struct wst(abs(img.at<uchar>(x, y) - img.at<uchar>(x + 1, y)), x + 1, y, marker);
            WS_que.push(wst);
        }
    }
    if (y - 1 >= 0) {//add up
        if (!tag.at<uchar>(x, y - 1)) {
            WS_struct wst(abs(img.at<uchar>(x, y) - img.at<uchar>(x, y - 1)), x, y - 1, marker);
            WS_que.push(wst);
        }
    }

    if (y + 1 < img.rows) {//add down
        if (!tag.at<uchar>(x, y + 1)) {
            WS_struct wst(abs(img.at<uchar>(x, y) - img.at<uchar>(x, y + 1)), x, y + 1, marker);
            WS_que.push(wst);
        }
    }
}



cv::Mat watershed(cv::Mat img_gray, cv::Mat tag, std::priority_queue<WS_struct> WS_que) {
    //int grad = 0;
    //int i = 0;
    while (!WS_que.empty()) {//S에 있는 주변부 확장
        WS_struct wst = WS_que.top(); //class
        WS_que.pop();
        int x = wst.x;
        int y = wst.y;
        int marker = wst.marker;
        //i++;
        /*
        if ((grad != wst.dist)) {
            if (i > 20000) {
                cv::namedWindow("watersheding");
                cv::imshow("watersheding", tag);
                cv::waitKey(0);
                grad = wst.dist;
                std::cout << grad << std::endl;
                i = 0;
            }
        }*/
        if (!tag.at<uchar>(x, y)) { //미정 tag marker로 마킹
            tag.at<uchar>(x, y) = marker;
            add_neighbor(WS_que, marker, x, y, img_gray, tag);

        }
    }

    return tag;
}



std::vector<std::vector<int>> point;
std::vector<std::vector<int>> color;
std::vector<int> tmpColor;
cv::Mat img;
cv::Mat img_gray;
cv::Mat mask;
cv::Point ptOld;
void on_mouse(int event, int x, int y, int flags, void*);
int b, g, r;
int thickness;
int state = 1;
int rand() {
    // 시드값을 얻기 위한 random_device 생성.
    std::random_device rd;

    // random_device 를 통해 난수 생성 엔진을 초기화 한다.
    std::mt19937 gen(rd());

    // 0 부터 255 까지 균등하게 나타나는 난수열을 생성하기 위해 균등 분포 정의.
    std::uniform_int_distribution<int> dis(0, 255);


    return dis(gen);
}


int findMask() {
    if (mask.empty()) {
        return -1;
    }
    std::vector<int>pt;
    for (int y = 0; y < mask.rows; y++)
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) == 255) {
                pt.push_back(x);
                pt.push_back(y);
                pt.push_back(state);
                point.push_back(pt);
                pt.clear();
                //std::cout << "eee" << std::endl;
            }
        }
    return 0;
}


void on_mouse(int event, int x, int y, int flags, void*)
{
    std::vector<int> pt;
    thickness = cv::getTrackbarPos("thickness", "img");
    if (thickness < 1) {
        thickness = 1;
    }
    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        ptOld = cv::Point(x, y);
        cv::circle(img, cv::Point(x, y), int(thickness / 2), cv::Scalar(b, g, r), -1);
        cv::circle(mask, cv::Point(x, y), int(thickness / 2), cv::Scalar(255), -1);

        cv::imshow("img", img);
        // cv::imshow("mask", mask);

         //pt.push_back(x);
         //pt.push_back(y);
         //pt.push_back(state);
         //point.push_back(pt);
         //std::cout << point.back()[0] << "," << point.back()[1] << std::endl;

        break;
    case cv::EVENT_LBUTTONUP:

        break;
    case cv::EVENT_RBUTTONDOWN:

        findMask();
        mask = 0;
        b = rand();
        g = rand();
        r = rand();
        tmpColor.push_back(b);
        tmpColor.push_back(g);
        tmpColor.push_back(r);
        color.push_back(tmpColor);
        tmpColor.clear();
        state++;
        std::cout << state << "번째 마커" << std::endl;


    case cv::EVENT_MOUSEMOVE:
        if (flags & cv::EVENT_FLAG_LBUTTON) {
            cv::line(img, ptOld, cv::Point(x, y), cv::Scalar(b, g, r), thickness);
            cv::line(mask, ptOld, cv::Point(x, y), cv::Scalar(255), thickness);
            cv::imshow("img", img);
            //cv::imshow("mask", mask);
            ptOld = cv::Point(x, y);
        }
        break;
    default:
        break;
    }
}

void onChange(int value, void* userdata) {
}

int main(void) {
    b = rand();
    g = rand();
    r = rand();
    tmpColor.push_back(b);
    tmpColor.push_back(g);
    tmpColor.push_back(r);
    color.push_back(tmpColor);
    tmpColor.clear();
    img = cv::imread("sample.jpg", 1);
    cv::fastNlMeansDenoisingColored(img, img);

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    
    mask = img_gray.clone();
    mask = 0;
    std::cout << state << "번째 마커" << std::endl;
    //cv::namedWindow("mask");
    cv::namedWindow("img");
    if (img.empty()) {

        return -1;
    }
    cv::createTrackbar("thickness", "img", 0, 20, onChange);
    cv::setMouseCallback("img", on_mouse);

    cv::imshow("img", img);
    cv::waitKey(0);
    findMask();
    /*cv::Mat img_grad;
    int morph_size = 2;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2 * morph_size + 1, 2 * morph_size + 1),cv::Point(morph_size, morph_size));
    cv::morphologyEx(img_gray, img_grad,cv::MORPH_GRADIENT, element,cv:: Point(-1, -1), 1);
    cv::namedWindow("diff");
    cv::imshow("diff", img_grad);*/

    cv::Mat tag = cv::Mat::zeros(img_gray.rows, img_gray.cols, CV_8UC1);

    std::priority_queue<WS_struct> WS_que;

    /*
    WS_struct gray(img_grad.at<float>(175, 175), 175, 175, 125);
    WS_que.push(gray);
    WS_struct black(img_grad.at<float>(300, 300), 300, 300, 255);
    WS_que.push(black);
    WS_struct white(img_grad.at<float>(20, 20), 20, 20, 1);
    WS_que.push(white);*/
    for (int i = 0; i < point.size(); i++) {
        WS_struct gray(img_gray.at<float>(point[i][1], point[i][0]), point[i][1], point[i][0], point[i][2]);
        WS_que.push(gray);

        //std::cout << "(" << point[i][0] << "," << point[i][1] << ") state:" << point[i][2] << std::endl;
    }

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    watershed(img_gray, tag, WS_que);
    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    std::cout << "Time cost(sec): " << sec.count() << "seconds" << std::endl;
    //cv::addWeighted(img_gray, 0.5, tag, 0.5, 0, tag);
    cv::Mat tagImg = img.clone();
    tagImg = 0;

    if (color.size()) {

        for (int i = 1; i <= state; i++) {
            for (int y = 0; y < img_gray.rows; y++) {
                for (int x = 0; x < img_gray.cols; x++) {
                    if (tag.at<uchar>(y, x) == i) {
                        int j = i - 1;
                        tagImg.at<cv::Vec3b>(y, x)[0] = color[j][0];
                        tagImg.at<cv::Vec3b>(y, x)[1] = color[j][1];
                        tagImg.at<cv::Vec3b>(y, x)[2] = color[j][2];
                        //std::cout<<"state" <<i<<":"<< x << "," << y << std::endl;
                    }
                }
            }
        }
    }

    cv::namedWindow("Input Image");
    cv::imshow("Input Image", img_gray);
    cv::namedWindow("Output Image");

    cv::imshow("Output Image", tagImg);



    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;

}
