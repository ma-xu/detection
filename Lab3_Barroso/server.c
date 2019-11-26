/*
 * Author:         Ricardo Barroso
 * Date:           11/5/19
 * Intructor:      Dr. Qing Yang
 * Description:    A server that mimic a TCP 3-way handshake and TCP connection closing.
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>


/*
 * Name:        tcp_hdr (Struct)
 * Description: A struct used to represent the header of a TCP Segment
 */
struct tcp_hdr {
    unsigned short int  src;            /* An unsigned short integer used to represent the 16-bit source port of the TCP Segment */
    unsigned short int  des;            /* An insigned short integer used to represent the 16-bit destination port of the TCP Segment */
    unsigned int        seq;            /* An unsigned integer used to represent the sequence number of the TCP Segment */
    unsigned int        ack;            /* An unsigned integer used to represent the acknowledgement number of the TCP Segment */
    unsigned short int  hdr_flags;      /* An unsigned short integer used to represent data offset/header length, the reserved section, and any flags of the TCP Segment */
    unsigned short int  rec;            /* An unsigned short integer used to represent the receive window for flow control of the TCP Segment */
    unsigned short int  cksum;          /* An unsigned short integer used to represent the Internet checksum value of the TCP Segment */
    unsigned short int  ptr;            /* An unsigned short integer used to represent the Urgent data pointer of the TCP Segment */
    unsigned int        opt;            /* An unsigned integer used to represent the options of the TCP segment of the TCP Segment */
};


/*
 * Name:           populate (Function)
 * Parameters:     N/A
 * Return:         A struct of type tcp_hdr
 * Description:    This function sets all of the fields of the tcp_hdr struct object to a default value
 */
struct tcp_hdr populate()
{
    /* Variables */
    struct tcp_hdr temp; /* A struct of type tcp_hdr used to assigned default values to the desired struct */

    /* Block of code that assigns values to the struct object atrributes */
    temp.src    = 0;
    temp.des    = 0;
    temp.seq    = 0;
    temp.ack    = 0;
    temp.hdr_flags  = 0;
    temp.rec    = 0;
    temp.cksum  = 0;
    temp.ptr    = 0;
    temp.opt    = 0;

    return(temp); /* Returns the struct object to populate the target struct object */
}


/*
 * Name:           reassign (Function)
 * Paramaeters:    A char array used to represent that tramsission TCP Segment
 * Return;         A struct of type tcp_hdr
 * Description:    This function tokenizes a char array and uses the tokens to reconstruct a TCP segment
 */
struct tcp_hdr reassign( char *a )
{
    /* Variables */
    struct tcp_hdr      temp;                                                                           /* A struct object of type tcp_hdr */
    char            tempSource[32], tempDestination[32], tempSequence[32], tempAcknowledgement[32]; /* Char arrays used to temporarily hold strings for conversion */
    char            tempFlags[32], tempReceiving[32], tempChecksum[32], tempPointer[32], tempOptions[32];
    unsigned short int  src;                                                                            /* An unsigned short integer used to represent the 16-bit source port of the TCP Segment */
    unsigned short int  des;                                                                            /* An insigned short integer used to represent the 16-bit destination port of the TCP Segment */
    unsigned int        seq;                                                                            /* An unsigned integer used to represent the sequence number of the TCP Segment */
    unsigned int        ack;                                                                            /* An unsigned integer used to represent the acknowledgement number of the TCP Segment */
    unsigned short int  hdr_flags;                                                                      /* An unsigned short integer used to represent data offset/header length, the reserved section, and any flags of the TCP Segment */
    unsigned short int  rec;                                                                            /* An unsigned short integer used to represent the receive window for flow control of the TCP Segment */
    unsigned short int  cksum;                                                                          /* An unsigned short integer used to represent the Internet checksum value of the TCP Segment */
    unsigned short int  ptr;                                                                            /* An unsigned short integer used to represent the Urgent data pointer of the TCP Segment */
    unsigned int        opt;                                                                            /* An unsigned integer used to represent the options of the TCP segment of the TCP Segment */

    /* Block of code that splits the string and reformats and assigns the segments */
    sscanf( a, "%s%s%s%s%s%s%s%s%s", tempSource, tempDestination, tempSequence, tempAcknowledgement, tempFlags, tempReceiving, tempChecksum, tempPointer, tempOptions );
    src     = atoi( tempSource );
    des     = atoi( tempDestination );
    seq     = atoi( tempSequence );
    ack     = atoi( tempAcknowledgement );
    hdr_flags   = atoi( tempFlags );
    rec     = atoi( tempReceiving );
    cksum       = atoi( tempChecksum );
    ptr     = atoi( tempPointer );
    opt     = atoi( tempOptions );
    temp.src    = src;
    temp.des    = des;
    temp.seq    = seq;
    temp.ack    = ack;
    temp.hdr_flags  = hdr_flags;
    temp.rec    = rec;
    temp.cksum  = cksum;
    temp.ptr    = ptr;
    temp.opt    = opt;

    return(temp);
}


/*
 * Name:           checkSum (Function)
 * Parameters:     A struct object of type tcp_hdr representing the specified TCP Segment
 * Return:         An unsigned short integer used to represent the calculated checkSum value
 * Description:    This function uses the attribute values of the provided tcp_hdr struct object and calculates the corresponding checksum value
 */
unsigned int checkSum( struct tcp_hdr a )
{
    /* Variables */
    unsigned int        temp;
    unsigned short int  array[12];
    unsigned int        i;
    unsigned int        sum = 0;

    memcpy( array, &a, 24 );

    for ( i = 0; i < 12; i++ )
    {
        sum = sum + array[i];
    }

    temp    = sum >> 16;
    sum = sum & 0x0000FFFF;
    sum = temp + sum;

    temp    = sum >> 16;
    sum = sum & 0x0000FFFF;
    temp    = temp + sum;

    printf( "Checksum Value: 0x%04X\n", (0xFFFF ^ temp) );
    return(temp);
}


/*
 * Name:           printTCP (Function)
 * Paramaters:     A struct object of type tcp_hdr representing the specified TCP Segment
 * Return;         N/A
 * Description:    This function prints the contents of a tcp_hdr struct object
 */
void printTCP( struct tcp_hdr a )
{
    /* Block of code thar prints TCP Segment Sections */
    printf( "Source:         0x%04X\n", a.src );
    printf( "Destination:    0x%04X\n", a.des );
    printf( "Sequence:       0x%08X\n", a.seq );
    printf( "Acknowledgement:0x%08X\n", a.ack );
    printf( "Flags:          0x%04X\n", a.hdr_flags );
    printf( "Receiving:      0x%04X\n", a.rec );
    printf( "Checksum:       0x%04X\n", (0xFFFF ^ a.cksum) );
    printf( "Pointer:        0x%04X\n", a.ptr );
    printf( "Options:        0x%08X\n", a.opt );

    return;
}


/*
 * Name:           populateString (Function)
 * Parameters:  A struct object of type tcp_hdr representing the specified TCP Segment
 * Returns:        A char array used to represent the transmission string for the desired TCP Segment
 * Description:    This function converts and stores the values of a tcp_hdr struct object in char arrays.
 *              The arrays are then combined to form one long transmission string.
 */
char* populateString( struct tcp_hdr a )
{
    /* Variables */
    char *tempArray;                                                                                        /* A char pointer used to return the content of the function */
    tempArray = (char *) malloc( 1000 );
    char    requestSource[32], requestDestination[32], requestSequence[32], requestAcknowledgement[32];     /* Char arrays used to store tcp attribute */
    char    requestFlags[32], requestReceiving[32], requestChecksum[32], requestPointer[32], requestOptions[32];

    /* Block of code that converts integers to strings */
    sprintf( requestSource, "%d", a.src );
    sprintf( requestDestination, "%d", a.des );
    sprintf( requestSequence, "%d", a.seq );
    sprintf( requestAcknowledgement, "%d", a.ack );
    sprintf( requestFlags, "%d", a.hdr_flags );
    sprintf( requestReceiving, "%d", a.rec );
    sprintf( requestChecksum, "%d", a.cksum );
    sprintf( requestPointer, "%d", a.ptr );
    sprintf( requestOptions, "%d", a.opt );

    /* Block of code that creates transmission string */
    strcpy( tempArray, requestSource );
    strcat( tempArray, " " );
    strcat( tempArray, requestDestination );
    strcat( tempArray, " " );
    strcat( tempArray, requestSequence );
    strcat( tempArray, " " );
    strcat( tempArray, requestAcknowledgement );
    strcat( tempArray, " " );
    strcat( tempArray, requestFlags );
    strcat( tempArray, " " );
    strcat( tempArray, requestReceiving );
    strcat( tempArray, " " );
    strcat( tempArray, requestChecksum );
    strcat( tempArray, " " );
    strcat( tempArray, requestPointer );
    strcat( tempArray, " " );
    strcat( tempArray, requestOptions );
    strcat( tempArray, " " );

    return(tempArray);                              /* Returns the transmission string */
}


int main( int argc, char **argv )
{
    /* Variables */
    int         port = atoi( argv[1] ); /* Integer used for the port number */
    char            str[1000];
    int         listen_fd, conn_fd;
    struct sockaddr_in  servaddr;

    /* AF_INET - IPv4 IP , Type of socket, protocol*/
    listen_fd = socket( AF_INET, SOCK_STREAM, 0 );

    bzero( &servaddr, sizeof(servaddr) );

    servaddr.sin_family     = AF_INET;
    servaddr.sin_addr.s_addr    = htons( INADDR_ANY );
    servaddr.sin_port       = htons( port );

    /* Binds the above details to the socket */
    bind( listen_fd, (struct sockaddr *) &servaddr, sizeof(servaddr) );

    /* Start listening to incoming connections */
    listen( listen_fd, 10 );

    /* Accepts an incoming connection */
    conn_fd = accept( listen_fd, (struct sockaddr *) NULL, NULL );
    printf( "Connection established...\n\n" );
    read( conn_fd, str, sizeof(str) );                      /* Reads the TCP Segement sent by the client */
    struct tcp_hdr receivedRequestTCP;                      /* A struct object of type tcp_hdr used to represent the connection request TCP segment sent by the client */
    receivedRequestTCP = reassign( str );                   /* Assigns values to receivedRequestTCP */
    printf( "The following Request TCP segement was received...\n" );
    printTCP( receivedRequestTCP );                         /* Calls the printTCP function the print the request connection TCP segment infromation */

    /*
     * Block of code that generates connection granted TCP segment
     * Variables
     */
    struct tcp_hdr  grantingTCP;                            /* A struct object of type tcp_hdr used to represent the granted TCP segement sent from the server */
    char        *grantingArray;                         /* A char pointer used for the populate string function */
    char        holder[1000];                           /* A char array used to store the transmission strings */
    grantingTCP     = populate();                   /* Sets defaulst to grantingTCP */
    grantingTCP.seq     = receivedRequestTCP.seq + 1;   /* Sets sequence number to client's sequence number +1 */
    grantingTCP.ack     = receivedRequestTCP.seq + 1;   /* Sets acknowledgement number ot client's sequence number + 1 */
    grantingTCP.hdr_flags   = 6162;                         /* Sets header to 24 and SYN and ACK bits to 1 */
    printf( "\n" );
    grantingTCP.cksum = checkSum( grantingTCP );            /* Calls the checkSum function and assigns the results to cksum */
    printf( "The following connection granted TCP segment is ready to be sent...\n" );
    printTCP( grantingTCP );                                /* Calls the printTCP function to print grantingTCP information */
    grantingArray = populateString( grantingTCP );          /* Generate the transmission string for grantingTCP */
    strcpy( holder, grantingArray );                        /* Copies transmission string from char pointer to char array */
    write( conn_fd, holder, sizeof(holder) );               /* Write the TCP segement to the client */
    printf( "Connection Granted Segment Sent...Awaiting Response...\n\n" );

    /* Block of code that handles the received acknowledgement TCP segment */
    char holder3[1000];                                     /* Char array used to store stransmission strings */
    read( conn_fd, holder3, sizeof(holder3) );              /* Reads the TCP segment sent by the client */
    struct tcp_hdr receivedACK;                             /* A struct object of type tcp_hdr used to represent the acknowledgement TCP segment sent by the client */
    receivedACK = reassign( holder3 );                      /* Assigns values to receivedACK TCP segment */
    printf( "The following Acknowledgement TCP segment was received...\n" );
    printTCP( receivedACK );                                /* Calls the printTCP function to print the receivedTCP TCP segemnt information */

    /* Block of code that handles the received close request TCP segment */
    char holder2[1000];                                     /* Char array used to hold transmission strings */
    read( conn_fd, holder2, sizeof(holder2) );              /* Reads in the TCP segment sent by the client */
    struct tcp_hdr receivedClosing;                         /* A struct object of type tcp_hdr used to represent the close request TCP segment sent by the client */
    receivedClosing = reassign( holder2 );                  /* Assigns values to receivedCLosing TCP segment */
    printf( "\nThe following Close Request TCP segment was received...\n" );
    printTCP( receivedClosing );                            /* Calls the printTCP function to print the closing request TCP segment information */

    /* Block of code that sends the first close request acknowledgement TCP segment */
    struct tcp_hdr  closingACK;                             /* A struct object of type tcp_hdr used to represent the 1st acknowledgement TCP segment sent by the server */
    char        *cACKArray;                             /* A char pointer used for the populateString function */
    closingACK      = populate();                   /* Sets defualts to closingACK */
    closingACK.seq      = receivedClosing.seq + 1;      /* Sets the sequence number to the client's sequence number + 1 */
    closingACK.ack      = receivedClosing.seq + 1;      /* Sets the acknowledgement number to the client's sequence number + 1 */
    closingACK.hdr_flags    = 6160;                         /* Sets the header to 24 and the ACK bit to 1 */
    printf( "\n" );
    closingACK.cksum = checkSum( closingACK );              /* Calls the checkSum function and assigns the result to cksum */
    printf( "The following Close Request Acknowledgement Segment is ready to be sent...\n" );
    printTCP( closingACK );                                 /* Calls the printTCP function to print the TCP Segment information */
    cACKArray = populateString( closingACK );               /* Generates the transmission string for the first ack TCP segment */
    bzero( holder, sizeof(holder) );                        /* Clears holder (sets everything to zero) */
    strcpy( holder, cACKArray );                            /* Copies the transmission string from char pointer to char array */
    write( conn_fd, holder, sizeof(holder) );               /* Write the transmission strinf for the segment to the client */
    printf( "Close Request Acknowledgement Segment Sent...\n\n" );

    /* Block of code that sends the second close request acknowledgement TCP segment */
    struct tcp_hdr  closingACK2;                            /* A struct object of type tcp_hdr used to represent the 2nd acknowledgement TCP segment sent by the server */
    char        *cACKArray2;                            /* A char pointer used to for the populateString function */
    closingACK2     = populate();                   /* Sets defaults to closingACK2 */
    closingACK2.seq     = receivedClosing.seq + 1;      /* Sets sequence number to client's sequence number + 1 */
    closingACK2.ack     = receivedClosing.seq + 1;      /* Sets acknowledgement number to client's sequence number + 1 */
    closingACK2.hdr_flags   = 6145;                         /* Sets header to 24 and FIN flag to 1 */
    printf( "\n" );
    closingACK2.cksum = checkSum( closingACK2 );            /* Calls the checkSum function and assigns it's value to cksum */
    printf( "The following Second Close Request Acknowledgement Segment is ready to be sent...\n" );
    printTCP( closingACK2 );                                /* Calls the printTCP function to print the TCP Segment information */
    cACKArray2 = populateString( closingACK2 );             /* Generates the transmission string for the second ack TCP segement */
    bzero( holder, sizeof(holder) );                        /* Clears holder (sets everything to zero) */
    strcpy( holder, cACKArray2 );                           /* Copies transmission string from char pointer to char array */
    write( conn_fd, holder, sizeof(holder) );               /* Writes the transmission string for the segment to the client */
    printf( "Second Close Request Acknowledgement Segment Sent...\n\n" );

    /* Block of code that handles he received final acknowledgement TCP segment */
    struct tcp_hdr  finalTCP;                               /* A struct object of type tcp_hdr used to represent the final acknowledgement TCP segment */
    char        holder4[1000];                          /* A char array used as a buffer */
    read( conn_fd, holder4, sizeof(holder4) );              /* Reads in transmission string from client */
    finalTCP = reassign( holder4 );                         /* Assigns attributes to finalTCP */
    printf( "The following Final Acknowledgement TCP Segement was received...\n" );
    printTCP( finalTCP );                                   /* Calls the printTCP function to print the TCP Segment information */

    printf( "\nClosing Connection...\n" );
    close( conn_fd );                                       /* close the connection */
}


