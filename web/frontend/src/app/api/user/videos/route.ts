import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";

export async function GET(request: NextRequest) {
    try {
        const session = await auth();
        const userId = session?.user?.id || "default-user";

        const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
        const backendResponse = await fetch(
            `${backendUrl}/api/user/videos`,
            {
                method: "GET",
                headers: {
                    "user-id": userId,
                },
            },
        );

        if (!backendResponse.ok) {
            const errorData = await backendResponse.json();
            return NextResponse.json(
                { error: errorData.detail || "Failed to fetch user videos" },
                { status: backendResponse.status },
            );
        }

        const videos = await backendResponse.json();

        return NextResponse.json(videos);
    } catch (error) {
        console.error("Fetch user videos error:", error);
        return NextResponse.json(
            { error: "Failed to fetch user videos" },
            { status: 500 },
        );
    }
}
