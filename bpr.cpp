#include <bits/stdc++.h>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/thread.hpp>
using namespace std;

typedef long long LL;
typedef pair<int, int> Pii;

// random generator
typedef boost::minstd_rand base_generator_type;
base_generator_type generator(42u);
boost::uniform_real<> uni_dist(0, 1);
boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);

const int NUM_USERS = 6102;
const int NUM_ITEMS = 12082;
const int MAX_DIM = 1050;

struct Record
{
    vector<int> positiveItems;
    set<int> itemSet;
}rec[NUM_USERS + 1];

vector<Pii> sample;

set<int> imp_fb[NUM_USERS + 1];
float vecU[NUM_USERS + 1][MAX_DIM], vecI[NUM_ITEMS + 1][MAX_DIM];
int dim = 100, num_threads = 20;
LL total_samples = 2e7, current_sample_count = 0;

// regularization parameter
float lambdaU = 0.0025, lambdaI = 0.0025, lambdaJ = 0.00025;
float init_rho = 0.025, rho;

// read data and construct the pairwise
void ReadData()
{
    FILE *fin = fopen("ciao_train.txt", "r");
    //FILE *fin = fopen("ml_training.dat", "r");
    int source, target, edge;
    while (~fscanf(fin, "%d%d%d", &source, &target, &edge))
    {
        if (source <= NUM_USERS && target > NUM_USERS)
        {
            rec[source].positiveItems.push_back(target - NUM_USERS);
            rec[source].itemSet.insert(target - NUM_USERS);
            sample.push_back({source, target - NUM_USERS});
        }
    }
    fclose(fin);
}

// initialize vectors randomly
void InitVector()
{
    for (int i = 0; i < dim; i++)
    {
        for (int j = 1; j <= NUM_USERS; j++)
            vecU[j][i] = (rand() / (float)RAND_MAX - 0.5) / dim;
        for (int j = 1; j <= NUM_ITEMS; j++)
            vecI[j][i] = (rand() / (float)RAND_MAX - 0.5) / dim;
    }
}

void SampleTripe(int &u, int &i, int &j)
{
    /*
    int len = rec[u].positiveItems.size();
    i = rec[u].positiveItems[int(uni() * len)];
    len = NUM_ITEMS;
    */
    int len = sample.size();
    int idx = uni() * len;
    u = sample[idx].first;
    i = sample[idx].second;
    j = uni() * NUM_ITEMS + 1;
    while (rec[u].itemSet.find(j) != rec[u].itemSet.end())
        j = uni() * NUM_ITEMS + 1;
    //if (u <= 1) cout << u << ' ' << i << ' ' << j << endl;
}

void Update(int u, int i, int j)
{

    float gradientU[dim], gradientI[dim], gradientJ[dim];
    fill(gradientU, gradientU + dim, 0.0);
    fill(gradientI, gradientI + dim, 0.0);
    fill(gradientJ, gradientJ + dim, 0.0);

    // common term
    float uiPro = 0.0;
    float ujPro = 0.0;
    for (int d = 0; d < dim; d++)
    {
        uiPro += vecU[u][d] * vecI[i][d];
        ujPro += vecU[u][d] * vecI[j][d];
    }
    float f = 1.0 / (1.0 + exp(uiPro - ujPro));

    // update
    for (int d = 0; d < dim; d++)
    {
        gradientU[d] += lambdaU * vecU[u][d] + f * (vecI[j][d] - vecI[i][d]);
        gradientI[d] += lambdaI * vecI[i][d] - f * vecU[u][d];
        gradientJ[d] += lambdaJ * vecI[j][d] + f * vecU[u][d];

        vecU[u][d] -= rho * gradientU[d];
        vecI[i][d] -= rho * gradientI[d];
        vecI[j][d] -= rho * gradientJ[d];
    }
}

float Loss()
{
    int num_loss_samples = 100 * sqrt(NUM_USERS);
    float rank_loss = 0.0, reg_loss = 0.0;
    for (int i = 0; i < num_loss_samples; i++)
    {
        /*
        int u = uni() * NUM_USERS;
        while (rec[u].positiveItems.size() == 0)
            u = uni() * NUM_USERS;
        */
        int u, pos, neg;
        SampleTripe(u, pos, neg);
        float predict_loss = 0.0;
        for (int j = 0; j < dim; j++)
        {
            predict_loss += vecU[u][j] * (vecI[pos][j] - vecI[neg][j]);
            reg_loss += 0.5 * lambdaU * vecU[u][j] * vecU[u][j];
            reg_loss += 0.5 * lambdaI * vecI[pos][j] * vecI[pos][j];
            reg_loss += 0.5 * lambdaJ * vecI[neg][j] * vecI[neg][j];
        }
        rank_loss += 1.0 / (1.0 + exp(predict_loss));
    }
    return rank_loss + reg_loss;
}

int tempCount = 0;

void *Train(void *id)
{
    LL count = 0, last_count = 0;
    while (true)
    {
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count > 10000)
		{
			current_sample_count += count - last_count;
			last_count = count;

			float current_loss = Loss();
			printf("%cRho: %f  Progress: %.3lf%%\tLoss : %.3f", 13, rho, (float)current_sample_count / (float)(total_samples + 1) * 100, current_loss);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

        //for (int u = 1; u <= NUM_USERS; u++)
        for (int iter = 1; iter <= 8; iter++)
        {
            /*
            int u = uni() * NUM_USERS;
            if (rec[u].positiveItems.size() == 0)
                continue;
            */
            int u, i, j;
            SampleTripe(u, i, j);
            //if (tempCount < 10)
            //    cout << u << ' ' << i << ' ' << j << endl;
            tempCount++;
            Update(u, i, j);
            break;
        }

        count++;
    }
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

void Output()
{
    FILE *fout = fopen("ciao_vec.txt", "w+");
    //FILE *fout = fopen("ml_vec.txt", "w+");
    fprintf(fout, "%lld %lld\n", NUM_USERS + NUM_ITEMS, dim);
    for (int i = 1; i <= NUM_USERS + NUM_ITEMS; i++)
    {
        if (i <= NUM_USERS)
        {
            fprintf(fout, "%d", i);
            for (int j = 0; j < dim; j++)
                fprintf(fout, " %f", vecU[i][j]);
            fprintf(fout, "\n");
        }
        else
        {
            fprintf(fout, "%d", i);
            for (int j = 0; j < dim; j++)
                fprintf(fout, " %f", vecI[i - NUM_USERS][j]);
            fprintf(fout, "\n");
        }
    }
    fclose(fout);
}

int main(int argc, char **argv)
{
    ReadData();
    cout << "data loaded" << endl;
    InitVector();
    cout << "vector initialized" << endl;
    clock_t start = clock();
    rho = init_rho;

    boost::thread *pt = new boost::thread[num_threads];
    long a;
    for (a = 0; a < num_threads; a++) pt[a] = boost::thread(Train, (void *)a);
	for (a = 0; a < num_threads; a++) pt[a].join();
    cout << endl << "train finished" << endl;

    clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    Output();
    return 0;
}
