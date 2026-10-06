import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";

export async function GET(
    request: NextRequest,
    { params }: { params: Promise<{ videoId: string }> },
) {
    try {
        const session = await auth();
        const userId = session?.user?.id || "default-user";

        const { videoId } = await params;

        const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
        const backendResponse = await fetch(
            `${backendUrl}/api/download/${videoId}/status`,
            {
                method: "GET",
                headers: {
                    "user-id": userId,
                },
            },
        );

        if (!backendResponse.ok) {
            const errorData = await backendResponse.json().catch(() => ({}));
            return NextResponse.json(
                { error: errorData.detail || "Failed to get download status" },
                { status: backendResponse.status },
            );
        }

        const data = await backendResponse.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("Download status error:", error);
        return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
}
