# PDF Parser – Sequence Diagram

This document describes the end-to-end flow of the PDF image extraction system. The design applies **SOLID** principles:

- **S**ingle Responsibility: Each component has one clear responsibility (API, storage, queue, workers, U-Net).
- **O**pen/Closed: Workers and storage are extensible without changing core flow.
- **L**iskov Substitution: Image extractor workers are interchangeable.
- **I**nterface Segregation: Narrow interfaces (e.g. PDF receiver, queue consumer, zip poller).
- **D**ependency Inversion: Backend depends on abstractions (queue, storage, zip location) not concrete implementations.

---

## High-Level Flow

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant BackendAPI as Backend API
    participant Storage as PDF/Output Storage
    participant Queue as Request Queue
    participant Worker1 as Image Extractor Worker 1
    participant Worker2 as Image Extractor Worker 2
    participant UNetAPI as U-Net API

    Note over User,UNetAPI: Phase 1 – Ingest & Enqueue
    User->>+BackendAPI: POST /extract (PDF file)
    BackendAPI->>Storage: Save PDF to location (path_PDF)
    BackendAPI->>Queue: Enqueue {pdf_path, job_id, callback_zip_path}

    Note over User,UNetAPI: Phase 2 – Workers process (parallel)
    loop Poll queue
        Worker1->>Queue: Poll for job
        Worker2->>Queue: Poll for job
    end
    Queue-->>Worker1: Job {pdf_path, job_id}
    Worker1->>Storage: Fetch PDF
    Storage-->>Worker1: PDF bytes

    par Process pages in parallel
        Worker1->>Worker1: Page 1 → base64
        Worker1->>UNetAPI: Get mask (page 1)
        UNetAPI-->>Worker1: Mask
        Worker1->>Worker1: Extract images using mask
        Worker1->>Storage: Save images to dir(job_id)/page_1/
    and
        Worker1->>Worker1: Page 2 → base64
        Worker1->>UNetAPI: Get mask (page 2)
        UNetAPI-->>Worker1: Mask
        Worker1->>Worker1: Extract images using mask
        Worker1->>Storage: Save images to dir(job_id)/page_2/
    end

    Worker1->>Storage: All pages done → zip dir(job_id) → save zip at callback_zip_path

    Note over User,UNetAPI: Phase 3 – Backend holds connection, polls, and responds
    loop Every 3 seconds (until timeout)
        BackendAPI->>Storage: Check if zip exists at callback_zip_path
        Storage-->>BackendAPI: Not found / Found
    end

    alt Zip ready before timeout
        Storage-->>BackendAPI: Zip exists
        BackendAPI->>Storage: Fetch zip file
        Storage-->>BackendAPI: Zip bytes
        BackendAPI->>User: 202 OK (zip file)
    else Timeout reached
        BackendAPI->>User: 408 Request Timeout {error: "Processing timed out"}
    end
    deactivate BackendAPI
```

---

```

---

## Component Responsibilities (SOLID Mapping)

| Component            | Responsibility                          | SOLID note                    |
|---------------------|-----------------------------------------|-------------------------------|
| **Backend API**     | Receive PDF, enqueue job, hold connection, poll for zip, return zip or timeout error | SRP: orchestration only       |
| **PDF Storage**      | Save/retrieve PDF by path               | SRP; depend on interface      |
| **Request Queue**   | Enqueue/dequeue job descriptors         | SRP; backend & workers depend on abstraction |
| **Image Extractor Worker** | Poll queue, load PDF, call U-Net, extract & save images, create zip | SRP per worker; workers substitutable (LSP) |
| **U-Net API**       | Accept base64 page, return mask         | SRP; single interface         |
| **Image/Zip Storage** | Save images per job, write zip at path   | SRP; backend polls abstraction |

---

## Parallelism Summary

- **Multiple workers**: Several image-extractor jobs poll the same queue; each claimed job is processed by one worker.
- **Multiple pages**: Within one job, pages are processed in parallel (e.g. thread pool or async tasks); each page: base64 → U-Net mask → extract images → save to `dir(job_id)/page_N/`.
- **Backend polling**: Holds the user connection open and polls every 3 seconds until the zip appears or a timeout is reached; responds with 202 + zip file on success, or 408 timeout error on failure.
