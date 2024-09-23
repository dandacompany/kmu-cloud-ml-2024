-- Active: 1721203012068@@dante-pgsql.c1ams2w0aobe.ap-northeast-2.rds.amazonaws.com@5432@postgres@public

-- 테이블 생성
CREATE TABLE users(
    id int NOT NULL PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    create_time DATE,
    name VARCHAR(255),
    age int,
    sex VARCHAR(255)
);

-- 데이터 삽입
INSERT INTO users (create_time, name, age, sex) VALUES ('2022-01-01', 'dante', 25, 'male');

-- 데이터 조회
SELECT * FROM users;