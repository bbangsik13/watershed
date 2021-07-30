#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <cmath>
#include <queue>
#include <chrono>



/**************************[ queue 관련 ]*******************************/
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


/*************************[ add함수(queue push)구현 ]********************************/
void add_neighbor(std::priority_queue<WS_struct>& WS_que, int marker, int x, int y, cv::Mat img, cv::Mat tag) {
    if (x - 1 >= 0) {//add left
        if (!tag.at<uchar>(x - 1, y)) {//tag 안 되어 있으면 queue 넣기 (dist, x, y, marker)
            WS_struct wst(abs(img.at<uchar>(x, y)-img.at<uchar>(x - 1, y)), x - 1, y, marker);
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



/*********************************************************/

//tag 빈배열

//S priority queue          

cv::Mat watershed(cv::Mat img_gray, cv::Mat tag) {

    std::priority_queue<WS_struct> WS_que;


    //전처리..?

    //cv::GaussianBlur(img_grad, img_grad, cv::Size(3,3), 10.0);




    // 임시 초기 라벨링

    /**********/
    WS_struct black(float(0), 306, 306, 80);
    WS_que.push(black);
    

    WS_struct white(float(0), 30, 30, 255);
    WS_que.push(white);



    /**********/
    int test_cnt = 0;
    while (!WS_que.empty()) {//S에 있는 주변부 확장
        WS_struct wst = WS_que.top(); //class
        WS_que.pop();
        int x = wst.x;
        int y = wst.y;
        int marker = wst.marker;

        if (!tag.at<uchar>(x, y)) { //미정 tag marker로 마킹
            tag.at<uchar>(x, y) = marker;
            add_neighbor(WS_que, marker, x, y, img_gray, tag);
        }
        /*
        if (!(test_cnt++ % 1000)) {
            cv::namedWindow("test");
            cv::imshow("test", tag);
            cv::waitKey(0);
            cv::destroyWindow("test");
        }*/

    }

    return tag;
}

int main(void)
{
    cv::Mat img_gray = cv::imread("sample.jpg", cv::IMREAD_GRAYSCALE);
    cv::fastNlMeansDenoising(img_gray, img_gray, 10, 7, 21);
    cv::Mat tag = cv::Mat::zeros(img_gray.cols, img_gray.rows, CV_8UC1);


    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    watershed(img_gray, tag);

    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    std::cout << "Time cost(sec): " << sec.count() << "seconds" << std::endl;

    cv::namedWindow("Input Image");
    cv::imshow("Input Image", img_gray);
    cv::namedWindow("Output Image");
    cv::imshow("Output Image", tag);



    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;


}
