#include <string>
#include <iostream>

using namespace std;
class Chrono {
//private properties
	int _h, _m,_s; //hour:minute:second
	static int nb_object;//how many Chrono object use ?
//for calculs
	static const int MINUTE_PER_HOUR = 60;
	static const int SEC_PER_MINUTE = 60;
public:
//2 constructors with nb_object counter and overflow detection
	Chrono(int heure = 0, int minute = 0, int seconde = 0) :
		_h(heure), _m(minute), _s(seconde) {
		nb_object++;
		if (is_overflow()) {
			throw string("error overflow ");
		}
	}
	Chrono(const string heure) {
		nb_object++;
		*this = string_to_chrono(heure);
	}
//destructor for count down object counter
	~Chrono() {
		nb_object--;
	}

//add function
	Chrono& add_second(int value) {
		_s = (_s+value) % SEC_PER_MINUTE;
		_m = _m + (value / SEC_PER_MINUTE) % MINUTE_PER_HOUR;
		_h = _h + value / (MINUTE_PER_HOUR*SEC_PER_MINUTE);
		if (is_overflow())
			raz();
		return *this;
	}
	Chrono& add_minute(int value){
		_m += (_m+value)% MINUTE_PER_HOUR;
		_h =_h + _m / MINUTE_PER_HOUR;
		if (is_overflow())
			raz();
		return *this;
	}
	Chrono& add_heure(int value){
		_h += value;
		if (is_overflow())
			raz();
		return *this;
	}
//raz function
	void raz() {
		_h = _m = _s = 0;
	}
//operator functions
	Chrono& operator+=(const Chrono& heure) {
		(*this).add_second(heure.s()).add_minute(heure.m()).add_heure(heure.h());
		return *this;
	}
	Chrono& operator+=(const int second) {
		(*this).add_second(second);
		return *this;
	}
	Chrono& operator=(const string heure) {
		(*this) = string_to_chrono(heure);
		return *this;
	}
	bool operator==(const Chrono& heure) {
		return (_h == heure.h() && _m == heure.m() && _s == heure.s());
	}
//getter 
	int      s() const { return _s; }
	int       m() const { return _m; }
	int       h() const { return _h; }
	operator int () const {
		return _h * MINUTE_PER_HOUR*SEC_PER_MINUTE + _m * SEC_PER_MINUTE + _s;
	}
//static function
	static int nb_instance() {
		return nb_object;
	}
private :
	bool is_overflow() {
		return(_h > 23 || _m > MINUTE_PER_HOUR || _s > SEC_PER_MINUTE);
	}
// string to Chrono conversion with throw error
	Chrono& string_to_chrono(string heure) {
		int first_separator = heure.find(':');
		int second_separator = heure.rfind(':');
		if (first_separator > 2 || (heure.size() - second_separator) > 2)
			throw string("error format :") + heure;
		string h = heure.substr(0, first_separator);
		string m = heure.substr(first_separator + 1, 2);
		string s = heure.substr(second_separator + 1, heure.size());
		(*this)._h = stoi(h); (*this)._m = stoi(m); (*this)._s = stoi(s);
		if (is_overflow()) {
			throw string("error overflow ");
		}
		return *this;
	}
};
//function for use of cin or cout for example
ostream& operator<<(ostream& os, const Chrono& chrono){
	return os << chrono.h() << ":" << chrono.m() << ":" << chrono.s();
}
istream& operator>>(istream& is, Chrono& chrono){
	string s;
	is >> s;
	chrono = s; // conversion method from string to chrono
	return is;
}

int Chrono::nb_object = 0; //static variable need to be here !!!
//just to see the count up and count down with destructor
void test_function() {
	Chrono test;
	cout << "in test function nb instance : " << Chrono::nb_instance() << endl;
}
//test class Chrono 
int main() {
	//begin to test static counter of Chrono and use of destructor
	Chrono test;
	cout << "before test function nb instance : " << Chrono::nb_instance() << endl;
	test_function();
	cout << "after test function nb instance : " << Chrono::nb_instance() << endl;
	//test of function operator<<
	cout <<"first objet test : " << test << endl;
	cout << "enter an time : format(hh:mm:ss) -> ";
	try {
		cin >> test;//test of function operator>>
	}
	catch (string s) {//test of n operator= with throw i
		cout << s << endl;
		test = "0:0:0";
		cout << "raz chronometer test  :" << test << endl ;
	}
	Chrono test1("1:2:3");// test of Chrono(const string)
	Chrono test3("1:2:3");
	if (test1 == test3)// test of operator ==
		cout << "ok" << endl;
	int nb_seconds = test1;// test of operator int () const method
	cout << "1:2:3 = " << nb_seconds<< " seconds" << endl;
	try {
		test += test1;// test of operator+=(const Chrono &)
		test += 3601;// test of operator+=(int)
	}
	catch (string s) {
		cout << test << " "<< s << ":"<< "raz"<< endl;
		test.raz();
	}
	cout << test << endl;
	try {
		test.add_second(82861).add_minute(1).add_heure(1);// test of add function
	}
	catch (string s) {
		cout << test << " " << s << ":" << "raz" << endl;
		test.raz();
	}
	cout << test <<endl;
}
