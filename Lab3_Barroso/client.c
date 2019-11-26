/*
 * Author:         Ricardo Barroso
 * Date:           11/5/19
 * Intructor:      Dr. Qing Yang
 * Description:    A client that mimic a TCP 3-way handshake and TCP connection closing.
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>


/*
 * Name:           tcp_hdr (Struct)
 * Description:    A struct used to represent the header of a TCP Segment
 */
struct tcp_hdr {
	unsigned short int	src;            /* An unsigned short integer used to represent the 16-bit source port of the TCP Segment */
	unsigned short int	des;            /* An insigned short integer used to represent the 16-bit destination port of the TCP Segment */
	unsigned int		seq;            /* An unsigned integer used to represent the sequence number of the TCP Segment */
	unsigned int		ack;            /* An unsigned integer used to represent the acknowledgement number of the TCP Segment */
	unsigned short int	hdr_flags;      /* An unsigned short integer used to represent data offset/header length, the reserved section, and any flags of the TCP Segment */
	unsigned short int	rec;            /* An unsigned short integer used to represent the receive window for flow control of the TCP Segment */
	unsigned short int	cksum;          /* An unsigned short integer used to represent the Internet checksum value of the TCP Segment */
	unsigned short int	ptr;            /* An unsigned short integer used to represent the Urgent data pointer of the TCP Segment */
	unsigned int		opt;            /* An unsigned integer used to represent the options of the TCP segment of the TCP Segment */
};


/*
 * Name:		populate (Function)
 * Parameters:	N/A
 * Return:		A struct of type tcp_hdr
 * Description:	This function sets all of the fields of the tcp_hdr struct object to a default value
 */
struct tcp_hdr populate()
{
	/* Variables */
	struct tcp_hdr temp; /* A struct of type tcp_hdr used to assigned default values to the desired struct */

	/* Block of code that assigns values to the struct object atrributes */
	temp.src	= 0;
	temp.des	= 0;
	temp.seq	= 0;
	temp.ack	= 0;
	temp.hdr_flags	= 0;
	temp.rec	= 0;
	temp.cksum	= 0;
	temp.ptr	= 0;
	temp.opt	= 0;

	return(temp); /* Returns the struct object to populate the target struct object */
}


/*
 * Name:		reassign (Function)
 * Paramaeters:	A char array used to represent that tramsission TCP Segment
 * Return;		A struct of type tcp_hdr
 * Description:	This function tokenizes a char array and uses the tokens to reconstruct a TCP segment
 */
struct tcp_hdr reassign( char *a )
{
	/* Variables */
	struct tcp_hdr		temp;                                                                           /* A struct object of type tcp_hdr */
	char			tempSource[32], tempDestination[32], tempSequence[32], tempAcknowledgement[32]; /* Char arrays used to remporarily holder strings for conversion */
	char			tempFlags[32], tempReceiving[32], tempChecksum[32], tempPointer[32], tempOptions[32];
	unsigned short int	src;                                                                            /* An unsigned short integer used to represent the 16-bit source port of the TCP Segment */
	unsigned short int	des;                                                                            /* An insigned short integer used to represent the 16-bit destination port of the TCP Segment */
	unsigned int		seq;                                                                            /* An unsigned integer used to represent the sequence number of the TCP Segment */
	unsigned int		ack;                                                                            /* An unsigned integer used to represent the acknowledgement number of the TCP Segment */
	unsigned short int	hdr_flags;                                                                      /* An unsigned short integer used to represent data offset/header length, the reserved section, and any flags of the TCP Segment */
	unsigned short int	rec;                                                                            /* An unsigned short integer used to represent the receive window for flow control of the TCP Segment */
	unsigned short int	cksum;                                                                          /* An unsigned short integer used to represent the Internet checksum value of the TCP Segment */
	unsigned short int	ptr;                                                                            /* An unsigned short integer used to represent the Urgent data pointer of the TCP Segment */
	unsigned int		opt;                                                                            /* An unsigned integer used to represent the options of the TCP segment of the TCP Segment */

	/* Block of code that splits the string and reformats and assigns the segments */
	sscanf( a, "%s%s%s%s%s%s%s%s%s", tempSource, tempDestination, tempSequence, tempAcknowledgement, tempFlags, tempReceiving, tempChecksum, tempPointer, tempOptions );
	src		= atoi( tempSource );
	des		= atoi( tempDestination );
	seq		= atoi( tempSequence );
	ack		= atoi( tempAcknowledgement );
	hdr_flags	= atoi( tempFlags );
	rec		= atoi( tempReceiving );
	cksum		= atoi( tempChecksum );
	ptr		= atoi( tempPointer );
	opt		= atoi( tempOptions );
	temp.src	= src;
	temp.des	= des;
	temp.seq	= seq;
	temp.ack	= ack;
	temp.hdr_flags	= hdr_flags;
	temp.rec	= rec;
	temp.cksum	= cksum;
	temp.ptr	= ptr;
	temp.opt	= opt;

	return(temp);
}


/*
 * Name:		checkSum (Function)
 * Parameters:	A struct object of type tcp_hdr representing the specified TCP Segment
 * Return:		An unsigned short integer used to represent the calculated checkSum value
 * Description:	This function uses the attribute values of the provided tcp_hdr struct object and calculates the corresponding checksum value
 */
unsigned int checkSum( struct tcp_hdr a )
{
	/* Variables */
	unsigned int		temp;
	unsigned short int	array[12];
	unsigned int		i;
	unsigned int		sum = 0;

	memcpy( array, &a, 24 );

	for ( i = 0; i < 12; i++ )
	{
		sum = sum + array[i];
	}

	temp	= sum >> 16;
	sum	= sum & 0x0000FFFF;
	sum	= temp + sum;

	temp	= sum >> 16;
	sum	= sum & 0x0000FFFF;
	temp	= temp + sum;

	printf( "Checksum Value: 0x%04X\n", (0xFFFF ^ temp) );
	return(temp);
}


/*
 * Name:		printTCP (Function)
 * Paramaters:	A struct object of type tcp_hdr representing the specified TCP Segment
 * Return;		N/A
 * Description:	This function prints the contents of a tcp_hdr struct object
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
 * Name:		populateString (Function)
 * Parameters:	A struct object of type tcp_hdr representing the specified TCP Segment
 * Returns:	A char array used to represent the transmission string for the desired TCP Segment
 * Description:	This function converts and stores the values of a tcp_hdr struct object in char arrays.
 * The arrays are then combined to form one long transmission string.
 */
char* populateString( struct tcp_hdr a )
{
	/* Vairables */
	char *tempArray;                                                                                        /* A char pointer used to retrun the content of the function */
	tempArray = (char *) malloc( 1000 );
	char	requestSource[32], requestDestination[32], requestSequence[32], requestAcknowledgement[32];     /* Char arrays used to store TCP attributes */
	char	requestFlags[32], requestReceiving[32], requestChecksum[32], requestPointer[32], requestOptions[32];

	/* Block of code used to convert integers to strings */
	sprintf( requestSource, "%d", a.src );
	sprintf( requestDestination, "%d", a.des );
	sprintf( requestSequence, "%d", a.seq );
	sprintf( requestAcknowledgement, "%d", a.ack );
	sprintf( requestFlags, "%d", a.hdr_flags );
	sprintf( requestReceiving, "%d", a.rec );
	sprintf( requestChecksum, "%d", a.cksum );
	sprintf( requestPointer, "%d", a.ptr );
	sprintf( requestOptions, "%d", a.opt );

	/* Block of code that creates the transmsiion string */
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

	return(tempArray);
}


int main( int argc, char **argv )
{
	/* Variables */
	int			port = atoi( argv[1] ); /* Integer used for the port number */
	int			sockfd, n;
	int			len = sizeof(struct sockaddr);
	char			recvline[40960];
	struct sockaddr_in	servaddr;

	/* AF_INET - IPv4 IP , Type of socket, protocol*/
	sockfd = socket( AF_INET, SOCK_STREAM, 0 );
	bzero( &servaddr, sizeof(servaddr) );

	servaddr.sin_family	= AF_INET;
	servaddr.sin_port	= htons( port ); /* Server port number */

	/* Convert IPv4 and IPv6 addresses from text to binary form */
	inet_pton( AF_INET, "129.120.151.94", &(servaddr.sin_addr) );

	/* Connect to the server */
	connect( sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr) );

	/*
	 * Block of code that send the requesting TCP Segment
	 * Variables
	 */
	struct tcp_hdr	requestingTCP;                          /* A struct object of type tcp_hdr used to represent the requesting TCP Segement */
	char		*requestingArray;                       /* Char array of size 10000 used to store the contents of the requesting TCP Segment */
	char		holder [1000];                          /* A char array used to hold strings for transmission */
	requestingTCP		= populate();                   /* Calls the populate function to assign deffaul values to each field */
	requestingTCP.hdr_flags = 6146;                         /* SYN bit is set to 1, sets header to 24 */
	requestingTCP.cksum	= checkSum( requestingTCP );    /* Calls to checkSum function and assigns its value to cksum */
	printf( "The following request TCP segment is ready to be sent...\n" );
	printTCP( requestingTCP );                              /* Calls the printTCP function to print the contents of the requesting TCP segment */
	requestingArray = populateString( requestingTCP );      /* Calls the populateString function to generate the transmission string */
	strcpy( holder, requestingArray );                      /* Copies transmission string from char pointer to char array */
	write( sockfd, holder, sizeof(holder) );                /* Sends the TCP Segment string to the server */
	printf( "Request Segment Sent...Awaiting Response...\n\n" );

	/* Block of code that recieves connection granted TCP Segment */
	bzero( holder, sizeof(holder) );                        /* Clears holder (sets everything to zero) */
	read( sockfd, holder, sizeof(holder) );                 /* Reads in from the server */
	struct tcp_hdr receivedGrantingTCP;                     /* A struct object of type tcp_hdr used to represent the connection granted TCP segment sent by the server */
	receivedGrantingTCP = reassign( holder );               /* Assigns values to TCP segment */
	printf( "The following Connection Granted TCP segment was received...\n" );
	printTCP( receivedGrantingTCP );                        /* Calls the printTCP funciton to print connection granted TCP segement information */
	printf( "\n" );

	/*
	 * Block of code that sends the Acknowledgement TCP segment
	 * Variables
	 */
	struct tcp_hdr	ackTCP;                                 /* A struct object of type tcp_hdr used to represent the acknowledgement TCP segement sent by the client */
	char		*ackArray;                              /* A char pointer used for the populateString function */
	ackTCP			= populate();                   /* Assings defaults to TCP segment */
	ackTCP.seq		= receivedGrantingTCP.seq + 1;  /* Sets squence number to receivedGranting's sequence number + 1 */
	ackTCP.ack		= receivedGrantingTCP.ack + 1;  /* Sets acknowledgement number to receivedGranting's sequence number + 1 */
	ackTCP.hdr_flags	= 6160;                         /* Sets header to 24 and sets ACK bit to 1 */
	ackTCP.cksum		= checkSum( ackTCP );           /* Calls the checksum function and assigns the results to cksum */
	printf( "The following acknowledgement TCP segment is ready to be sent...\n" );
	printTCP( ackTCP );                                     /* Calls the printTCP function to print the acknowledgement TCP segment's information */
	ackArray = populateString( ackTCP );                    /* Generates transmission string */
	bzero( holder, sizeof(holder) );                        /* Clears holder (sets to zero) */
	strcpy( holder, ackArray );                             /* Copies transmission string from char pointer to char array */
	write( sockfd, holder, sizeof(holder) );                /* Write the TCP segment to the server */
	printf( "Acknowledgement Segment Sent... \n\n" );

	/*
	 * Block of code that sends the Close Request TCP segment
	 * Variables
	 */
	struct tcp_hdr	closingTCP;                             /* A struct object of type tcp_hdr used to represent the closing request TCP segment sent by the client */
	char		*closingArray;                          /* A char pointer used for the populateString function */
	closingTCP		= populate();                   /* Sets defaults to closingTCP */
	closingTCP.hdr_flags	= 6145;                         /* Sets header to 24 and FIN bit to 1 */
	closingTCP.cksum	= checkSum( closingTCP );       /* Calls the checkSum function and assings results to cksum */
	printf( "The following close request TCP segement is ready to be sent...\n" );
	printTCP( closingTCP );                                 /* Calls the printTCP function to  print the closing request TCP segment's information */
	closingArray = populateString( closingTCP );            /* Generates the transmission string */
	bzero( holder, sizeof(holder) );                        /* Clears holder (sets everything to zero) */
	strcpy( holder, closingArray );                         /* Copies the transmission string from char pointer to char array */
	write( sockfd, holder, sizeof(holder) );                /* Send the TCP segment to the server */
	printf( "Close Request Segment Sent... \n\n" );

	/*
	 * Block of code that recieves the close request acknowledgement TCP segment
	 * Variables
	 */
	bzero( holder, sizeof(holder) );                        /* Clears holder (sets everything to zero) */
	read( sockfd, holder, sizeof(holder) );                 /* Reads in TCP segement from server */
	struct tcp_hdr receivingClosingACK;                     /* A struct object of type tcp_hdr used to represent the first closing acknowledgement TCP segment sent by the server */
	receivingClosingACK = reassign( holder );               /* Assings values to TCP segment */
	printf( "\nThe following close request acknowledgement TCP segment was received...\n" );
	printTCP( receivingClosingACK );                        /* Calls the printTCP function to print the first closing acknowledgement TCP segment's information */

	/*
	 * Block of code that recieves the second close request acknowledgment TCP segment
	 * Variables
	 */
	bzero( holder, sizeof(holder) );                        /* Clears holder (sets everything to zero) */
	read( sockfd, holder, sizeof(holder) );                 /* Reads in TCP segment form server */
	struct tcp_hdr receivingClosingACK2;                    /* A struct object of type tcp_hdr used to represent the second closing acknowledgement TCP segment sent by the server */
	receivingClosingACK2 = reassign( holder );              /* Assigns calues to TCP segment */
	printf( "\nThe following close request acknowledgement TCP segment was received...\n" );
	printTCP( receivingClosingACK2 );                       /* Calls the printTCP function to print the second closing acknowledgement TCP segment's information */
	printf( "\n" );

	/*
	 * Block of code that send the final acknowledgement TCP segment
	 * Variables
	 */
	struct tcp_hdr	finalTCP;                               /* A struct object of type tcp_hdr used to represent the final ackowledgement TCP segement */
	char		*finalArray;                            /* A char pointer used for populateString function */
	finalTCP		= populate();                   /* Sets defaults to finalTCP */
	finalTCP.seq		= closingTCP.seq + 1;           /* Sets sequence number to initial client's + 1 */
	finalTCP.ack		= receivingClosingACK.ack + 1;  /* Sets acknowledgement number to initial server's + 1 */
	finalTCP.hdr_flags	= 6160;                         /* Set header to 24 and ACK bit to 1 */
	finalTCP.cksum		= checkSum( finalTCP );         /* Calls the checkSum function and assigns the result to cksum */
	printf( "\nThe following final acknowledgement TCP segment is ready to be sent...\n" );
	printTCP( finalTCP );                                   /* Calls the printTCP function to print the final acknowledgement TCP segment's information */
	finalArray = populateString( finalTCP );                /* Generate the transmission string */
	bzero( holder, sizeof(holder) );                        /* Clears holder (sets everything to zero) */
	strcpy( holder, finalArray );                           /* Copies the transmission string from char pointer to char array */
	write( sockfd, holder, sizeof(holder) );                /* Sends the TCP segment to the server */
	printf( "Final Acknowledgement Segment Sent...\n" );

	printf( "\nConnection Closed...\n" );
}


