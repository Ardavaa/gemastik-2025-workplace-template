CREATE DATABASE Report;
Use Report;
CREATE TABLE eror (
    id_eror INT PRIMARY KEY AUTO_INCREMENT,  
    nama_eror VARCHAR(255) NOT NULL,         
    deskripsi_eror TEXT,                     
    isHandle BOOLEAN NOT NULL DEFAULT FALSE, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
);
CREATE TABLE User_isue (
    id_isue INT PRIMARY KEY AUTO_INCREMENT,    
    nama_isue VARCHAR(255) NOT NULL,           
    deskripsi_isue TEXT,                       
    isHandle BOOLEAN NOT NULL DEFAULT FALSE,   
    createrored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);