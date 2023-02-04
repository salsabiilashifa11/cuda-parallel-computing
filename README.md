# **CUDA**

## Semester II Tahun 2021/2022

### Tugas Besar 2 IF3230 Sistem Paralel dan Terdistribusi

_Program Studi Teknik Informatika_ <br />
_Sekolah Teknik Elektro dan Informatika_ <br />
_Institut Teknologi Bandung_ <br />

_Semester II Tahun 2021/2022_

## **Description**

Program paralel yang menerima satu matriks kernel dan _n_ buah matriks masukan. Program akan melakukan operasi konvolusi matriks pada matriks kernel dan setiap matriks masukan untuk menghasilkan matriks hasil konvolusi. Setelah itu, program Anda akan membuat _n_ bilangan bulat positif yang merupakan selisih elemen terbesar dan terkecil dari setiap matriks hasil konvolusi, serta melakukan sorting dari _n_ bilangan tersebut. <br />
<br/>
Keluaran program Anda adalah nilai maksimum, nilai minimum, median, dan rata-rata _n_ bilangan tersebut. Nilai median dan rata-rata dihitung menggunakan integer division untuk pembulatan. <br /><br />
Pengujian dilakukan pada WSL (Lokal) dan Google Colab pada [link](https://colab.research.google.com/drive/1t-HD1Cncq9Y4ejY3QDYMOyrpbdQWO2cf?usp=sharing) berikut.

## **Author**

1. Daru Bagus Dananjaya (13519080)
2. Aulia Adila (13519100)
3. Shifa Salsabiila (13519106)

## **Requirements**

- [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive)

## **Analisis Program**

**1. Jelaskan cara kerja program Anda, terutama pada paralelisasi dengan CUDA yang Anda implementasikan berdasarkan skema di atas.**<br>
Implementasi program konvolusi matriks dengan menggunakan CUDA diawali dengan insialisasi ukuran matriks kernel, jumlah matriks target, ukuran matriks target, serta penerimaan nilai untuk masing-masing matriks yang akan digunakan. Selanjutnya, dilakukan proses pengalokasian memori untuk proses yang akan dilakukan pada device. Proses alokasi memori ini dilakukan dengan menggunakan fugsi cudaMalloc() dan cudaMemcpy().<br>

Selanjutnya, dilakukan pemanggilan fungsi cuda_convolution untuk melakukan perhitungan konvolusi matriks antara setiap matriks target dengan matriks kernel. Bagian ini dilakukan di device, dengan memanfaatkan grid, block, serta thread untuk membuat proses perhitungan ini menjadi paralel. Setelah perhitungan konvolusi untuk setiap matriks selesai, maka dicari data range untuk masing-masing hasil yang masih dilakukan pada device.

Selanjutnya, hasil data range yang didapatkan dari perhitungan sebelumnya dipindahkan kembali ke host dan diurutkan dengan menggunakan fungsi merge sort. Proses pengurutan ini dilakukan pada device dengan memanfaatkan grid, block, serta thread yang ada. Setelah berhasil diurutkan, dicari nilai median dan rata-rata dari hasil.

**2. Dari waktu eksekusi terbaik program paralel Anda, bandingkan dengan waktu eksekusi program sekuensial yang diberikan. Analisis mengapa waktu eksekusi program Anda bisa lebih lambat / lebih cepat / sama saja. Lalu simpulkan bagaimana CUDA memengaruhi waktu eksekusi program Anda.**<br>
Kelompok kami membuat eksperimen melalui Google Colab dan WSL (Lokal), dengan waktu eksekusi yang diperoleh adalah sebagai berikut.

_Waktu eksekusi dengan Google Colab:_
Test Case | Serial | Paralel |
------------ | :----: | ------: |
TC1 | 12.621 ms | 30.065 ms |
TC2 | 754.792 ms | 978.531 ms |
TC3 | 722.748 ms | 227.321 ms |
TC4 | 9942.322 ms | 766.269 ms |

_Waktu eksekusi dengan WSL(Lokal):_
Test Case | Serial | Paralel |
------------ | :----: | ------: |
TC1 | 14.220 ms | 39.444 ms |
TC2 | 1266.037 ms | 218.036 ms |
TC3 | 1078.624 ms | 159.748 ms |
TC4 | 16705.729 ms | 1870.55 ms |

TC1 memiliki waktu eksekusi serial yang lebih cepat dibandingkan eksekusi paralel, karena pada program paralel terdapat overhead untuk melakukan preparasi (mempersiapkan grid, blok, dan thread).

TC2 dalam pengujian WSL (lokal) memiliki waktu eksekusi paralel yang lebih cepat dibandingkan waktu eksekusi serial namun tidak pada Google Colab. Penyebab dari hal ini tidak dapat diketahui karena secara general, skema paralelisasi yang kami buat telah menghasilkan waktu eksekusi yang lebih baik dibandingkan eksekusi secara serial

Jika dilihat dari keselarasan waktu eksekusi pada Google Colab dan WSL, TC4 dengan paralelisasi CUDA memiliki perbedaan waktu eksekusi yang paling signifikan, yaitu sebesar _766.269ms_ pada pengujian Google Colab (lebih cepat **130%** ms dibandingkan waktu eksekusi serial) dan _1870.55ms_ sekon pada pengujian WSL (lebih cepat **88%** ms daripada serial).

CUDA dapat memengaruhi waktu eksekusi dengan cara menyebar objek yang diproses melalui grid, block, dan thread yang berbeda. Setiap thread akan melakukan eksekusi program masing-masing, sehingga proses dapat berjalan secara paralel, yang berakibat pada waktu eksekusi program menjadi lebih cepat.

**3. Jelaskan secara singkat apakah ada perbedaan antara hasil keluaran program serial dan program paralel Anda, dan jika ada jelaskan juga penyebab dari perbedaan tersebut.**<br>
Berdasarkan hasil percobaan yang dilakukan, **tidak ada perbedaan antara hasil program serial dan paralel**. Keempat test case yang diujikan selalu menghasilkan summary yang sama. Perbedaan yang ada hanyalah pada waktu eksekusi program.

**4. Dengan paralelisasi yang Anda implementasikan, untuk bagian perhitungan konvolusi saja, dari 3 kasus berikut yang manakah yang waktu eksekusinya paling cepat dan mengapa?**<br>
a. Jumlah Matrix: 10000, Ukuran Kernel: 1x1, Ukuran Matrix: 1x1<br>
b. Jumlah Matrix: 1, Ukuran Kernel: 1x1, Ukuran Matrix: 100x100<br>
c. Jumlah Matrix: 1, Ukuran Kernel: 100x100, Ukuran Matrix: 100x100<br>
(Note: ketiga kasus memiliki jumlah operasi perkalian yang sama)<br>

Berdasarkan percobaan yang telah dilakukan,

Secara teori, **kasus a** akan menghasilkan waktu eksekusi paling cepat. Semakin besar ukuran dari sebuah matriks, maka akan mengakibatkan jumlah komputasi konvolusi matriks yang semakin banyak pula. Semakin banyak komputasi konvolusi matriks yang dilakukan, maka akan mengakibatkan memperlambat waktu eksekusi.<br>

Skema paralelisasi yang kami buat membuat sebuah thread melakukan satu buah perhitungan konvolusi matriks. Oleh karena itu, akan dialokasiskan 10000 thread kemudian melakukan perhitungan konvolusi untuk matriks input dan kernel yang masing-masing berukuran 1x1 Sehingga pada akhirnya, kasus ini akan menghasilkan waktu eksekusi yang lebih cepat dibandingkan kedua kasus lainnya.
