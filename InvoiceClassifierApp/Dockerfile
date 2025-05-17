# Base for running the app
FROM mcr.microsoft.com/dotnet/runtime:9.0 AS base
WORKDIR /app

# Build stage
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY ["InvoiceClassifierApp/InvoiceClassifierApp.csproj", "InvoiceClassifierApp/"]
RUN dotnet restore "InvoiceClassifierApp/InvoiceClassifierApp.csproj"
COPY . .
WORKDIR "/src/InvoiceClassifierApp"
RUN dotnet build "InvoiceClassifierApp.csproj" -c Release -o /app/build

# Publish stage
FROM build AS publish
RUN dotnet publish "InvoiceClassifierApp.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Final stage with Python
FROM mcr.microsoft.com/dotnet/runtime:9.0
WORKDIR /app
COPY --from=publish /app/publish .

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r /app/PythonProject2/requirements.txt

# Optional: run both .NET and Python in background
CMD dotnet InvoiceClassifierApp.dll
