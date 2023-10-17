FROM haskell

WORKDIR /app

COPY . .

RUN stack build

CMD ["stack", "run"]
