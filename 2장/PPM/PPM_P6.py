# 필요한 패키지를 import함
import array

class PPM_P6:
	def __init__(self):
		# 초기자를 정의하여 데이터 필드 초기화
		self.width = 0
		self.height = 0
		self.maxval = 0

	def __repr__(self):
		# 클래스 정보를 표현하는 문자열을 생성하여 반환
		return "PPM(\n'P6'\n {}, {}\n {}\n)".format(
			self.width, self.height, self.maxval)

	def read(self, filename):
		infile = open(filename, 'rb')

		# 헤더 정보 읽음
		idx = 0
		for line in infile:
			# 주석문은 처리 생략
			# 행의 첫 번째 문자가 '#' (0x23)인 경우 주석문으로 처리
			if(line[0] == 0x23):
				continue

			# 행 단위로 파일에서 데이터를 읽어 line 변수에 저장
			# 데이터의 마지막에는 줄바꿈 문자를 포함하고 있음
			# 줄바꿈 문자를 제거한 후
			# 파일에서 읽은 데이터만을 포함하는 list 생성
			# line 변수에 저장된 데이터는 bytes 자료형을 가짐
			line = line.splitlines()

			# Bytes 자료형을 문자열로 변환
			line = line[0].decode()

			# 공백 문자에 의한 문자열 분리
			line = line.split()

			# 헤더 정보 처리
			# idx는 식별자, 가로 길이, 세로 길이, 최대밝기를
			# 읽을 때마다 1씩 증가
			# 헤더 영역에 포함된 정보는 ASCII로 구성되어 있음
			# 구분자가 space bar 또는 tab 문자를 사용한 경우도 함께 고려함
			for token in line:
				if(idx == 0):
					# 식별자를 확인하여 'P6'이 아닌 경우
					# 에러 메시지를 출력하고 종료
					assert token == 'P6', 'Wrong filetype'
				elif(idx == 1):
					# 가로 길이 저장
					width = int(token)
				elif(idx == 2):
					# 세로 길이 저장
					height = int(token)
				elif(idx == 3):
					# 최대밝기 저장
					maxval = int(token)
					assert maxval < 256, 'Too many colors'

				idx = idx + 1

			if idx == 4: # 데이터 영역
				break

		# 비트맵 데이터 읽음
		bitmap = infile.read(width*height*3)

		# 파일 닫음
		infile.close()

		# 헤더 정보를 데이터 필드에 저장
		self.width = width
		self.height = height
		self.maxval = maxval

		# 결과 데이터 반환
		return (width, height, maxval, bitmap)

	def write(self, width, height, maxval, bitmap, filename):
		ppm_header = 'P6\n' + str(width) + '\n' + \
			str(height) + '\n' + str(maxval) + '\n'

		image = array.array('B', bitmap)
		# Save the PPM image as a binary file
		with open(filename, 'wb') as f:
			f.write(bytearray(ppm_header, 'ascii'))
			image.tofile(f)