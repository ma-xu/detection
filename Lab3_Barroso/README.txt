/*
Ricardo Barroso
11/5/19
Dr. Qing Yang
CSCE 3530
Lab 3
*/

Files: client.c, server.c

To complile:
	server.c
		gcc -o server server.c
		
	client.c:
		gcc -o client client.c
		
To run:
	server
		./server <port_number>
		
	client
		./client <port_number>
		
	Notes:
		server must be complied on CSE01.cse.unt.edu.
		client must be complied on CSE02.cse.unt.edu.
		The port number entered upon execution must be idetical for both server and client.

Description:
	The program will mimic the following:
		A 3-way TCP Handhsake between a server and a client:
			Upon running both the client and server, the client will generate a connection request TCP segment and send it to the server which will respond with a connection granted TCP segment.
			The client will respond with an acknowledgement TCP segment and send it to the server. The infomration in the TCP segment header wil be displayed before it is sent and after it is received.
		
		The closing of a TCP connection between a server and a client
			The client will generate a closing request TCP segment and send it to the server. The server will respond by sending two individual closing acknowledgement TCP segments. The client will respond
			by sending a final acknowledgement TCP segment to the server beofre the connection is closed. The information in the TCP segment header will be displayed before it is sent and after it is received.
Notes:
	The majority of the functionality for the code is in the utility functions created at the top.
		
	The sequence and acknowledgement numbers are set as they are described in the instructions. If a number is set differently than expected, it is because the instructions were unclear.

	The hdr_flags attribute of each TCP segment is set by assigning the corresponding decimal number.