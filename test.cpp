#include<bits/stdc++.h>
#define optimize ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL)
#define INF 0x3f3f3f3f
#define INFLL 0x3f3f3f3f3f3f3f3f
#define pii pair<int, int>
#define PI 3.141592653589793238462643383279502884L
#define mod % 1000000007
#define all(v) v.begin(), v.end()
#define ms(x, y) memset(x, y, sizeof(x))

using namespace std;

int main(){
    optimize;

    const int nnz = 10, nnx = 11, NUM_VARS = 3, hs = 2;

    for (int ll=0; ll<NUM_VARS; ll++) {
    for (int k=0; k<nnz; k++) {
      for (int i=0; i<nnx; i++) {
        int inds = (k+hs)*(nnx+2*hs) + ll*(nnz+2*hs)*(nnx+2*hs) + i+hs;
        int indt = ll*nnz*nnx + k*nnx + i;
        cout << inds - indt << "\n";
      }
    }
  }
    
    return 0;   
}
