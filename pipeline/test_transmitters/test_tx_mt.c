#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <time.h>
#include <errno.h>

#include <pthread.h>

#define PORT (10000)

//#define SERVERADDRESS "127.0.0.1"
#define SERVERADDRESS "100.100.101.100"


#define NCHAN 96
#define NCHAN_TOT (NCHAN*2)
#define NPOL 64
#define NPOL_TOT 704

// Packet delay is implemented every 16 * nchan_block packets (per thread)
// one packet is NCHAN*NPOL bytes
// -> delay occurs every 1.5Mbits
// -> 100,000 ns delay ~>15 Gb/s (per thread)
// -> 1,000,000 ns delay ~>17 Gb/s (total)
#define PACKET_DELAY_NS 10000

pthread_barrier_t barrier;

struct snap2_hdr_type {
        uint64_t  seq;       // Spectra counter == packet counter
        uint32_t  magic;     // = 0xaabbccdd
        uint16_t  npol;      // Number of polarizations in this packet
        uint16_t  npol_tot;      // Total number of polarizations for this pipeline
        uint16_t  nchan;     // Number of channels in this packet
        uint16_t  nchan_tot;     // Total number of channels for this pipeline
        uint32_t  chan_block_id; // Channel block ID. Eg. 0 for chans 0..nchan-1, 1 for chans nchan..2*nchan-1, etc.
        uint32_t  chan0;     // First channel in this packet 
        uint32_t  pol0;      // First pol in this packet 
};

struct packet {
    struct snap2_hdr_type header;
    char data[NCHAN*NPOL];
};

int nchan_blocks = NCHAN_TOT / NCHAN;
int npol_blocks = NPOL_TOT / NPOL;

int create_sockets(int source_port) {
    int sockfd;
    struct sockaddr_in client;

    printf("Configure socketi (sending from port %d)...\n", source_port);
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
    {
        fprintf(stderr, "Error opening socket");
        return EXIT_FAILURE;
    }

    bzero((char*)&client, sizeof(client));
    client.sin_family = AF_INET;
    client.sin_addr.s_addr = htonl(INADDR_ANY);
    client.sin_port = htons(source_port);
    bind(sockfd, (struct sockaddr *)&client, sizeof(client));
    return sockfd;
}

struct data {
    int sockfd;
    struct packet *pkt;
    struct sockaddr_in server;
    int pol;
    int delay;
};

void send_packets(void *args) {
    int i = 0;
    struct data *d = args;
    int chan_block_id;
    uint64_t bytes_sent = 0;
    struct timespec start, end, delay;
    delay.tv_sec = d->delay / 1000000000;
    delay.tv_nsec = d->delay % 1000000000;
    // Static header entries
    d->pkt->header.magic = htobe32((unsigned)time(NULL));
    d->pkt->header.npol = htobe16(NPOL);
    d->pkt->header.npol_tot = htobe16(NPOL_TOT);
    d->pkt->header.nchan = htobe16(NCHAN);
    d->pkt->header.nchan_tot = htobe16(NCHAN_TOT);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    while(1) {
        d->pkt->header.seq = htobe64(i);
        for (chan_block_id=0; chan_block_id<nchan_blocks; chan_block_id++) {
            d->pkt->header.chan_block_id = htobe32(chan_block_id);
            d->pkt->header.chan0 = htobe32(chan_block_id * NCHAN);
            if (sendto(d->sockfd, (void *)(d->pkt), sizeof(struct packet), 0,
                       (const struct sockaddr*)&(d->server), sizeof(d->server)) < 0)
            {
                fprintf(stderr, "Error in sendto()\n");
            }
        }
        if (i % 16 == 0) {
            nanosleep(&delay, NULL);
        }
        bytes_sent += nchan_blocks * sizeof(struct packet);
        i++;
        pthread_barrier_wait(&barrier);
        // Only let the pol 0 thread print stats
        if (d->pol == 0) {
            if (i % 10000 == 0) {
                bytes_sent = bytes_sent * npol_blocks; // scale by number of threads
                clock_gettime(CLOCK_MONOTONIC_RAW, &end);
			          uint64_t delta_ns = (end.tv_sec - start.tv_sec) * 1000000000 +
				    				(end.tv_nsec - start.tv_nsec);
                clock_gettime(CLOCK_MONOTONIC_RAW, &start);
                printf("Sent %ld bytes at %.2lf Gb/s\n", bytes_sent, (double)(8*bytes_sent) / delta_ns);
                bytes_sent = 0;
            }
        }
    }
}



int main(int argc, char **argv)
{
    int sockfd[npol_blocks];

    int delay = PACKET_DELAY_NS;
    if(argc > 1) {
        delay = atoi(argv[1]);
    }
    fprintf(stdout, "Inter-packet delay (per thread): %d ns\n", delay);

    struct sockaddr_in server;
    bzero((char*)&server, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr(SERVERADDRESS);
    server.sin_port = htons(PORT);
    
    struct packet pkt[npol_blocks];

    int chan_block_id, pol;

    printf("pol blocks: %d\n", npol_blocks);
    printf("chan blocks: %d\n", nchan_blocks);

    pthread_t threads[npol_blocks];
    
    pthread_barrier_init(&barrier, NULL, npol_blocks);

    struct data args[npol_blocks];
    for(pol=0; pol<npol_blocks; pol++) {
        args[pol].sockfd = create_sockets(PORT+1+pol);
        args[pol].pol = pol;
        args[pol].server = server;
        args[pol].pkt = pkt + pol;
        args[pol].delay = delay;
    }

    printf("Send UDP data...\n");

    for (pol=0; pol<npol_blocks; pol++) {
        if(pthread_create(&threads[pol], NULL, (void *)&send_packets, (void *)(&args[pol]))) {
                    fprintf(stderr, "Failed to create thread %d\n", pol);
        }
    }

    for (pol=0; pol<npol_blocks; pol++) {
        pthread_join(threads[pol], NULL);
    }

    pthread_barrier_destroy(&barrier);
    pthread_exit(0);
}
