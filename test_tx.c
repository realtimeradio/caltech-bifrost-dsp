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

#define PORT (10000)

#define SERVERADDRESS "127.0.0.1"
//#define SERVERADDRESS "100.100.100.101"


#define NCHAN 96
#define NCHAN_TOT (NCHAN*2)
#define NPOL 64
#define NPOL_TOT 704

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


int main(int argc, char **argv)
{
    struct timespec start, end;
    int sockfd;
    struct sockaddr_in server;
    
    struct packet pkt;

    printf("Configure socket...\n");
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
    {
        fprintf(stderr, "Error opening socket");
        return EXIT_FAILURE;
    }

    bzero((char*)&server, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr(SERVERADDRESS);
    server.sin_port = htons(PORT);

    int chan_block_id, pol;

    printf("Send UDP data...\n");
    int i = 0;
    uint64_t bytes_sent = 0;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //static entries
    while(1) {
        for (chan_block_id=0; chan_block_id<nchan_blocks; chan_block_id++) {
            for (pol=0; pol<npol_blocks; pol++) {
                pkt.header.magic = htobe32(0xaabbccdd);
                pkt.header.npol = htobe16(NPOL);
                pkt.header.npol_tot = htobe16(NPOL_TOT);
                pkt.header.nchan = htobe16(NCHAN);
                pkt.header.nchan_tot = htobe16(NCHAN_TOT);
                pkt.header.seq = htobe64(i);
                pkt.header.chan_block_id = htobe32(chan_block_id);
                pkt.header.chan0 = htobe32(chan_block_id * NCHAN);
                pkt.header.pol0 = htobe32(pol * NPOL);
                if (sendto(sockfd, &pkt, sizeof(pkt), 0,
                           (const struct sockaddr*)&server, sizeof(server)) < 0)
                {
                    fprintf(stderr, "Error in sendto()\n");
                    return EXIT_FAILURE;
                }
                bytes_sent += sizeof(pkt.data);
            }
        }
        if (i % 10000 == 0) {
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
			uint64_t delta_ns = (end.tv_sec - start.tv_sec) * 1000000000 +
								(end.tv_nsec - start.tv_nsec);
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            printf("Sent %ld bytes at %.2lf Gb/s\n", bytes_sent, (double)(8*bytes_sent) / delta_ns);
            bytes_sent = 0;
        }
        i++;
        //sleep(0.001);
    }

    return EXIT_SUCCESS;
}
