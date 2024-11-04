
template <class T>
class point
{
    T x,y;
};

template <class T>
T diff(const point<T> &p1, const point<T> &p2)
{
    return pow(p1.x-p2.x,2) + pow(p1.y-p2.y,2);
}

template <class T>
void move(point<T> &p ,const point<T> &offset)
{
    p.x+=offset.x;
    p.y+=offset.y;
}

template <class T>
point<T> operator*(const T &mul,const point<T> &p)
{

}

